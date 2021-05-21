##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###########################################################################################################################################
#
# In this file, we aims to evaluate three kinds of channel searching strategies:
# - channel-wise interpolation from "Network Pruning via Transformable Architecture Search, NeurIPS 2019"
# - masking + Gumbel-Softmax (mask_gumbel) from "FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions, CVPR 2020"
# - masking + sampling (mask_rl) from "Can Weight Sharing Outperform Random Architecture Search? An Investigation With TuNAS, CVPR 2020"
#
# For simplicity, we use tas, mask_gumbel, and mask_rl to refer these three strategies. Their official implementations are at the following links:
# - TAS: https://github.com/D-X-Y/AutoDL-Projects/blob/main/docs/NeurIPS-2019-TAS.md
# - FBNetV2: https://github.com/facebookresearch/mobile-vision
# - TuNAS: https://github.com/google-research/google-research/tree/master/tunas
####
# python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo mask_rl --arch_weight_decay 0 --warmup_ratio 0.25
####
# python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo tas --rand_seed 777
# python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo tas --rand_seed 777
# python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo tas --rand_seed 777
####
# python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo mask_gumbel --rand_seed 777
# python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo mask_gumbel --rand_seed 777
# python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo mask_gumbel --rand_seed 777
####
# python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo mask_rl --arch_weight_decay 0 --rand_seed 777 --use_api 0
# python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo mask_rl --arch_weight_decay 0 --rand_seed 777
# python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo mask_rl --arch_weight_decay 0 --rand_seed 777
###########################################################################################################################################
import os, sys, time, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import count_parameters_in_MB, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net, get_search_spaces
from nats_bench import create


# Ad-hoc for RL algorithms.
class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = 0
        self._denominator = 0
        self._momentum = momentum

    def update(self, value):
        self._numerator = (
            self._momentum * self._numerator + (1 - self._momentum) * value
        )
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    @property
    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


RL_BASELINE_EMA = ExponentialMovingAverage(0.95)


def search_func(
    xloader,
    network,
    criterion,
    scheduler,
    w_optimizer,
    a_optimizer,
    enable_controller,
    algo,
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
        base_inputs = base_inputs.cuda(non_blocking=True)
        arch_inputs = arch_inputs.cuda(non_blocking=True)
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        # Update the weights
        network.zero_grad()
        _, logits, _ = network(base_inputs)
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
        network.zero_grad()
        a_optimizer.zero_grad()
        _, logits, log_probs = network(arch_inputs)
        arch_prec1, arch_prec5 = obtain_accuracy(
            logits.data, arch_targets.data, topk=(1, 5)
        )
        if algo == "mask_rl":
            with torch.no_grad():
                RL_BASELINE_EMA.update(arch_prec1.item())
                rl_advantage = arch_prec1 - RL_BASELINE_EMA.value
            rl_log_prob = sum(log_probs)
            arch_loss = -rl_advantage * rl_log_prob
        elif algo == "tas" or algo == "mask_gumbel":
            arch_loss = criterion(logits, arch_targets)
        else:
            raise ValueError("invalid algorightm name: {:}".format(algo))
        if enable_controller:
            arch_loss.backward()
            a_optimizer.step()
        # record
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
    return (
        base_losses.avg,
        base_top1.avg,
        base_top5.avg,
        arch_losses.avg,
        arch_top1.avg,
        arch_top5.avg,
    )


def valid_func(xloader, network, criterion, logger):
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
            _, logits, _ = network(arch_inputs.cuda(non_blocking=True))
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
    # torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    train_data, valid_data, xshape, class_num = get_datasets(
        xargs.dataset, xargs.data_path, -1
    )
    if xargs.overwite_epochs is None:
        extra_info = {"class_num": class_num, "xshape": xshape}
    else:
        extra_info = {
            "class_num": class_num,
            "xshape": xshape,
            "epochs": xargs.overwite_epochs,
        }
    config = load_config(xargs.config_path, extra_info, logger)
    search_loader, train_loader, valid_loader = get_nas_search_loaders(
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

    search_space = get_search_spaces(xargs.search_space, "nats-bench")

    model_config = dict2config(
        dict(
            name="generic",
            super_type="search-shape",
            candidate_Cs=search_space["candidates"],
            max_num_Cs=search_space["numbers"],
            num_classes=class_num,
            genotype=args.genotype,
            affine=bool(xargs.affine),
            track_running_stats=bool(xargs.track_running_stats),
        ),
        None,
    )
    logger.log("search space : {:}".format(search_space))
    logger.log("model config : {:}".format(model_config))
    search_model = get_cell_based_tiny_net(model_config)
    search_model.set_algo(xargs.algo)
    logger.log("{:}".format(search_model))

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(
        search_model.weights, config
    )
    a_optimizer = torch.optim.Adam(
        search_model.alphas,
        lr=xargs.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=xargs.arch_weight_decay,
        eps=xargs.arch_eps,
    )
    logger.log("w-optimizer : {:}".format(w_optimizer))
    logger.log("a-optimizer : {:}".format(a_optimizer))
    logger.log("w-scheduler : {:}".format(w_scheduler))
    logger.log("criterion   : {:}".format(criterion))
    params = count_parameters_in_MB(search_model)
    logger.log("The parameters of the search model = {:.2f} MB".format(params))
    logger.log("search-space : {:}".format(search_space))
    if bool(xargs.use_api):
        api = create(None, "size", fast_mode=True, verbose=False)
    else:
        api = None
    logger.log("{:} create API = {:} done".format(time_string(), api))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )
    network, criterion = search_model.cuda(), criterion.cuda()  # use a single GPU

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )

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
        start_epoch, valid_accuracies, genotypes = 0, {"best": -1}, {-1: network.random}

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

        if (
            xargs.warmup_ratio is None
            or xargs.warmup_ratio <= float(epoch) / total_epoch
        ):
            enable_controller = True
            network.set_warmup_ratio(None)
        else:
            enable_controller = False
            network.set_warmup_ratio(
                1.0 - float(epoch) / total_epoch / xargs.warmup_ratio
            )

        logger.log(
            "\n[Search the {:}-th epoch] {:}, LR={:}, controller-warmup={:}, enable_controller={:}".format(
                epoch_str,
                need_time,
                min(w_scheduler.get_lr()),
                network.warmup_ratio,
                enable_controller,
            )
        )

        if xargs.algo == "mask_gumbel" or xargs.algo == "tas":
            network.set_tau(
                xargs.tau_max
                - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1)
            )
            logger.log("[RESET tau as : {:}]".format(network.tau))
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
            enable_controller,
            xargs.algo,
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

        genotype = network.genotype
        logger.log("[{:}] - [get_best_arch] : {:}".format(epoch_str, genotype))
        valid_a_loss, valid_a_top1, valid_a_top5 = valid_func(
            valid_loader, network, criterion, logger
        )
        logger.log(
            "[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}% | {:}".format(
                epoch_str, valid_a_loss, valid_a_top1, valid_a_top5, genotype
            )
        )
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
            logger.log("{:}".format(api.query_by_arch(genotypes[epoch], "90")))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    # the final post procedure : count the time
    start_time = time.time()
    genotype = network.genotype
    search_time.update(time.time() - start_time)

    valid_a_loss, valid_a_top1, valid_a_top5 = valid_func(
        valid_loader, network, criterion, logger
    )
    logger.log(
        "Last : the gentotype is : {:}, with the validation accuracy of {:.3f}%.".format(
            genotype, valid_a_top1
        )
    )

    logger.log("\n" + "-" * 100)
    # check the performance from the architecture dataset
    logger.log(
        "[{:}] run {:} epochs, cost {:.1f} s, last-geno is {:}.".format(
            xargs.algo, total_epoch, search_time.sum, genotype
        )
    )
    if api is not None:
        logger.log("{:}".format(api.query_by_arch(genotype, "90")))
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Weight sharing NAS methods to search for cells.")
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    parser.add_argument(
        "--search_space",
        type=str,
        default="sss",
        choices=["sss"],
        help="The search space name.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["tas", "mask_gumbel", "mask_rl"],
        help="The search space name.",
    )
    parser.add_argument(
        "--genotype",
        type=str,
        default="|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|",
        help="The genotype.",
    )
    parser.add_argument(
        "--use_api",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether use API or not (which will cost much memory).",
    )
    # FOR GDAS
    parser.add_argument(
        "--tau_min", type=float, default=0.1, help="The minimum tau for Gumbel Softmax."
    )
    parser.add_argument(
        "--tau_max", type=float, default=10, help="The maximum tau for Gumbel Softmax."
    )
    # FOR ALL
    parser.add_argument(
        "--warmup_ratio", type=float, help="The warmup ratio, if None, not use warmup."
    )
    #
    parser.add_argument(
        "--track_running_stats",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether use track_running_stats or not in the BN layer.",
    )
    parser.add_argument(
        "--affine",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether use affine=True or False in the BN layer.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/nas-benchmark/algos/weight-sharing.config",
        help="The path of configuration.",
    )
    parser.add_argument(
        "--overwite_epochs",
        type=int,
        help="The number of epochs to overwrite that value in config files.",
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
    parser.add_argument(
        "--arch_eps", type=float, default=1e-8, help="weight decay for arch encoding"
    )
    # log
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/search",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--print_freq", type=int, default=200, help="print frequency (default: 200)"
    )
    parser.add_argument("--rand_seed", type=int, help="manual seed")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    dirname = "{:}-affine{:}_BN{:}-AWD{:}-WARM{:}".format(
        args.algo,
        args.affine,
        args.track_running_stats,
        args.arch_weight_decay,
        args.warmup_ratio,
    )
    if args.overwite_epochs is not None:
        dirname = dirname + "-E{:}".format(args.overwite_epochs)
    args.save_dir = os.path.join(
        "{:}-{:}".format(args.save_dir, args.search_space), args.dataset, dirname
    )

    main(args)
