##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import sys, time, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import NASBench201API as API


def search_func(
    xloader,
    network,
    criterion,
    scheduler,
    w_optimizer,
    a_optimizer,
    epoch_str,
    print_freq,
    logger,
):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    network.train()
    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(
        xloader
    ):
        scheduler.update(None, 1.0 * step / len(xloader))
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        # update the weights
        sampled_arch = network.module.dync_genotype(True)
        network.module.set_cal_mode("dynamic", sampled_arch)
        # network.module.set_cal_mode( 'urs' )
        network.zero_grad()
        _, logits = network(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        w_optimizer.step()
        # record
        base_prec1, base_prec5 = obtain_accuracy(
            logits.data, base_targets.data, topk=(1, 5)
        )
        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))

        # update the architecture-weight
        network.module.set_cal_mode("joint")
        network.zero_grad()
        _, logits = network(arch_inputs)
        arch_loss = criterion(logits, arch_targets)
        arch_loss.backward()
        a_optimizer.step()
        # record
        arch_prec1, arch_prec5 = obtain_accuracy(
            logits.data, arch_targets.data, topk=(1, 5)
        )
        arch_losses.update(arch_loss.item(), arch_inputs.size(0))
        arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
        arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = (
                "*SEARCH* "
                + time_string()
                + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(xloader))
            )
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time
            )
            Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=base_losses, top1=base_top1, top5=base_top5
            )
            Astr = "Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=arch_losses, top1=arch_top1, top5=arch_top5
            )
            logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Astr)
            # print (nn.functional.softmax(network.module.arch_parameters, dim=-1))
            # print (network.module.arch_parameters)
    return (
        base_losses.avg,
        base_top1.avg,
        base_top5.avg,
        arch_losses.avg,
        arch_top1.avg,
        arch_top5.avg,
    )


def get_best_arch(xloader, network, n_samples):
    with torch.no_grad():
        network.eval()
        archs, valid_accs = network.module.return_topK(n_samples), []
        # print ('obtain the top-{:} architectures'.format(n_samples))
        loader_iter = iter(xloader)
        for i, sampled_arch in enumerate(archs):
            network.module.set_cal_mode("dynamic", sampled_arch)
            try:
                inputs, targets = next(loader_iter)
            except:
                loader_iter = iter(xloader)
                inputs, targets = next(loader_iter)

            _, logits = network(inputs)
            val_top1, val_top5 = obtain_accuracy(
                logits.cpu().data, targets.data, topk=(1, 5)
            )

            valid_accs.append(val_top1.item())

        best_idx = np.argmax(valid_accs)
        best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
        return best_arch, best_valid_acc


def valid_func(xloader, network, criterion):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    with torch.no_grad():
        network.eval()
        for step, (arch_inputs, arch_targets) in enumerate(xloader):
            arch_targets = arch_targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction
            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(
                logits.data, arch_targets.data, topk=(1, 5)
            )
            arch_losses.update(arch_loss.item(), arch_inputs.size(0))
            arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    train_data, valid_data, xshape, class_num = get_datasets(
        xargs.dataset, xargs.data_path, -1
    )
    config = load_config(
        xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger
    )
    search_loader, _, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        xargs.dataset,
        "configs/nas-benchmark/",
        (config.batch_size, config.test_batch_size),
        xargs.workers,
    )
    logger.log(
        "||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(
            xargs.dataset, len(search_loader), len(valid_loader), config.batch_size
        )
    )
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    search_space = get_search_spaces("cell", xargs.search_space_name)
    if xargs.model_config is None:
        model_config = dict2config(
            dict(
                name="SETN",
                C=xargs.channel,
                N=xargs.num_cells,
                max_nodes=xargs.max_nodes,
                num_classes=class_num,
                space=search_space,
                affine=False,
                track_running_stats=bool(xargs.track_running_stats),
            ),
            None,
        )
    else:
        model_config = load_config(
            xargs.model_config,
            dict(
                num_classes=class_num,
                space=search_space,
                affine=False,
                track_running_stats=bool(xargs.track_running_stats),
            ),
            None,
        )
    logger.log("search space : {:}".format(search_space))
    search_model = get_cell_based_tiny_net(model_config)

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(
        search_model.get_weights(), config
    )
    a_optimizer = torch.optim.Adam(
        search_model.get_alphas(),
        lr=xargs.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=xargs.arch_weight_decay,
    )
    logger.log("w-optimizer : {:}".format(w_optimizer))
    logger.log("a-optimizer : {:}".format(a_optimizer))
    logger.log("w-scheduler : {:}".format(w_scheduler))
    logger.log("criterion   : {:}".format(criterion))
    flop, param = get_model_infos(search_model, xshape)
    logger.log("FLOP = {:.2f} M, Params = {:.2f} MB".format(flop, param))
    logger.log("search-space : {:}".format(search_space))
    if xargs.arch_nas_dataset is None:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log("{:} create API = {:} done".format(time_string(), api))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )
    network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(last_info)
        )
        last_info = torch.load(last_info)
        start_epoch = last_info["epoch"]
        checkpoint = torch.load(last_info["last_checkpoint"])
        genotypes = checkpoint["genotypes"]
        valid_accuracies = checkpoint["valid_accuracies"]
        search_model.load_state_dict(checkpoint["search_model"])
        w_scheduler.load_state_dict(checkpoint["w_scheduler"])
        w_optimizer.load_state_dict(checkpoint["w_optimizer"])
        a_optimizer.load_state_dict(checkpoint["a_optimizer"])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                last_info, start_epoch
            )
        )
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        init_genotype, _ = get_best_arch(valid_loader, network, xargs.select_num)
        start_epoch, valid_accuracies, genotypes = 0, {"best": -1}, {-1: init_genotype}

    # start training
    start_time, search_time, epoch_time, total_epoch = (
        time.time(),
        AverageMeter(),
        AverageMeter(),
        config.epochs + config.warmup,
    )
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.val * (total_epoch - epoch), True)
        )
        epoch_str = "{:03d}-{:03d}".format(epoch, total_epoch)
        logger.log(
            "\n[Search the {:}-th epoch] {:}, LR={:}".format(
                epoch_str, need_time, min(w_scheduler.get_lr())
            )
        )

        (
            search_w_loss,
            search_w_top1,
            search_w_top5,
            search_a_loss,
            search_a_top1,
            search_a_top5,
        ) = search_func(
            search_loader,
            network,
            criterion,
            w_scheduler,
            w_optimizer,
            a_optimizer,
            epoch_str,
            xargs.print_freq,
            logger,
        )
        search_time.update(time.time() - start_time)
        logger.log(
            "[{:}] search [base] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s".format(
                epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum
            )
        )
        logger.log(
            "[{:}] search [arch] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
                epoch_str, search_a_loss, search_a_top1, search_a_top5
            )
        )

        genotype, temp_accuracy = get_best_arch(valid_loader, network, xargs.select_num)
        network.module.set_cal_mode("dynamic", genotype)
        valid_a_loss, valid_a_top1, valid_a_top5 = valid_func(
            valid_loader, network, criterion
        )
        logger.log(
            "[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}% | {:}".format(
                epoch_str, valid_a_loss, valid_a_top1, valid_a_top5, genotype
            )
        )
        # search_model.set_cal_mode('urs')
        # valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
        # logger.log('[{:}] URS---evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
        # search_model.set_cal_mode('joint')
        # valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
        # logger.log('[{:}] JOINT-evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
        # search_model.set_cal_mode('select')
        # valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
        # logger.log('[{:}] Selec-evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
        # check the best accuracy
        valid_accuracies[epoch] = valid_a_top1

        genotypes[epoch] = genotype
        logger.log(
            "<<<--->>> The {:}-th epoch : {:}".format(epoch_str, genotypes[epoch])
        )
        # save checkpoint
        save_path = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(xargs),
                "search_model": search_model.state_dict(),
                "w_optimizer": w_optimizer.state_dict(),
                "a_optimizer": a_optimizer.state_dict(),
                "w_scheduler": w_scheduler.state_dict(),
                "genotypes": genotypes,
                "valid_accuracies": valid_accuracies,
            },
            model_base_path,
            logger,
        )
        last_info = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(args),
                "last_checkpoint": save_path,
            },
            logger.path("info"),
            logger,
        )
        with torch.no_grad():
            logger.log("{:}".format(search_model.show_alphas()))
        if api is not None:
            logger.log("{:}".format(api.query_by_arch(genotypes[epoch], "200")))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    # the final post procedure : count the time
    start_time = time.time()
    genotype, temp_accuracy = get_best_arch(valid_loader, network, xargs.select_num)
    search_time.update(time.time() - start_time)
    network.module.set_cal_mode("dynamic", genotype)
    valid_a_loss, valid_a_top1, valid_a_top5 = valid_func(
        valid_loader, network, criterion
    )
    logger.log(
        "Last : the gentotype is : {:}, with the validation accuracy of {:.3f}%.".format(
            genotype, valid_a_top1
        )
    )

    logger.log("\n" + "-" * 100)
    # check the performance from the architecture dataset
    logger.log(
        "SETN : run {:} epochs, cost {:.1f} s, last-geno is {:}.".format(
            total_epoch, search_time.sum, genotype
        )
    )
    if api is not None:
        logger.log("{:}".format(api.query_by_arch(genotype, "200")))
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SETN")
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    # channels and number-of-cells
    parser.add_argument("--search_space_name", type=str, help="The search space name.")
    parser.add_argument("--max_nodes", type=int, help="The maximum number of nodes.")
    parser.add_argument("--channel", type=int, help="The number of channels.")
    parser.add_argument(
        "--num_cells", type=int, help="The number of cells in one stage."
    )
    parser.add_argument(
        "--select_num",
        type=int,
        help="The number of selected architectures to evaluate.",
    )
    parser.add_argument(
        "--track_running_stats",
        type=int,
        choices=[0, 1],
        help="Whether use track_running_stats or not in the BN layer.",
    )
    parser.add_argument(
        "--config_path", type=str, help="The path of the configuration."
    )
    # architecture leraning rate
    parser.add_argument(
        "--arch_learning_rate",
        type=float,
        default=3e-4,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    # log
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--save_dir", type=str, help="Folder to save checkpoints and log."
    )
    parser.add_argument(
        "--arch_nas_dataset",
        type=str,
        help="The path to load the architecture dataset (tiny-nas-benchmark).",
    )
    parser.add_argument("--print_freq", type=int, help="print frequency (default: 200)")
    parser.add_argument("--rand_seed", type=int, help="manual seed")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)
