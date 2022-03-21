##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v1 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v1 --drop_path_rate 0.3
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v1
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v2 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v2
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v2
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo gdas --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo gdas
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo gdas
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo setn --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo setn
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo setn
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo random
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo random
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
####
# The following scripts are added in 20 Mar 2022
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo gdas_v1 --rand_seed 777
######################################################################################
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


# The following three functions are used for DARTS-V2
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def _hessian_vector_product(
    vector, network, criterion, base_inputs, base_targets, r=1e-2
):
    R = r / _concat(vector).norm()
    for p, v in zip(network.weights, vector):
        p.data.add_(R, v)
    _, logits = network(base_inputs)
    loss = criterion(logits, base_targets)
    grads_p = torch.autograd.grad(loss, network.alphas)

    for p, v in zip(network.weights, vector):
        p.data.sub_(2 * R, v)
    _, logits = network(base_inputs)
    loss = criterion(logits, base_targets)
    grads_n = torch.autograd.grad(loss, network.alphas)

    for p, v in zip(network.weights, vector):
        p.data.add_(R, v)
    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


def backward_step_unrolled(
    network,
    criterion,
    base_inputs,
    base_targets,
    w_optimizer,
    arch_inputs,
    arch_targets,
):
    # _compute_unrolled_model
    _, logits = network(base_inputs)
    loss = criterion(logits, base_targets)
    LR, WD, momentum = (
        w_optimizer.param_groups[0]["lr"],
        w_optimizer.param_groups[0]["weight_decay"],
        w_optimizer.param_groups[0]["momentum"],
    )
    with torch.no_grad():
        theta = _concat(network.weights)
        try:
            moment = _concat(
                w_optimizer.state[v]["momentum_buffer"] for v in network.weights
            )
            moment = moment.mul_(momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, network.weights)) + WD * theta
        params = theta.sub(LR, moment + dtheta)
    unrolled_model = deepcopy(network)
    model_dict = unrolled_model.state_dict()
    new_params, offset = {}, 0
    for k, v in network.named_parameters():
        if "arch_parameters" in k:
            continue
        v_length = np.prod(v.size())
        new_params[k] = params[offset : offset + v_length].view(v.size())
        offset += v_length
    model_dict.update(new_params)
    unrolled_model.load_state_dict(model_dict)

    unrolled_model.zero_grad()
    _, unrolled_logits = unrolled_model(arch_inputs)
    unrolled_loss = criterion(unrolled_logits, arch_targets)
    unrolled_loss.backward()

    dalpha = unrolled_model.arch_parameters.grad
    vector = [v.grad.data for v in unrolled_model.weights]
    [implicit_grads] = _hessian_vector_product(
        vector, network, criterion, base_inputs, base_targets
    )

    dalpha.data.sub_(LR, implicit_grads.data)

    if network.arch_parameters.grad is None:
        network.arch_parameters.grad = deepcopy(dalpha)
    else:
        network.arch_parameters.grad.data.copy_(dalpha.data)
    return unrolled_loss.detach(), unrolled_logits.detach()


def search_func(
    xloader,
    network,
    criterion,
    scheduler,
    w_optimizer,
    a_optimizer,
    epoch_str,
    print_freq,
    algo,
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
        if algo == "setn":
            sampled_arch = network.dync_genotype(True)
            network.set_cal_mode("dynamic", sampled_arch)
        elif algo == "gdas":
            network.set_cal_mode("gdas", None)
        elif algo == "gdas_v1":
            network.set_cal_mode("gdas_v1", None)
        elif algo.startswith("darts"):
            network.set_cal_mode("joint", None)
        elif algo == "random":
            network.set_cal_mode("urs", None)
        elif algo == "enas":
            with torch.no_grad():
                network.controller.eval()
                _, _, sampled_arch = network.controller()
            network.set_cal_mode("dynamic", sampled_arch)
        else:
            raise ValueError("Invalid algo name : {:}".format(algo))

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
        if algo == "setn":
            network.set_cal_mode("joint")
        elif algo == "gdas":
            network.set_cal_mode("gdas", None)
        elif algo == "gdas_v1":
            network.set_cal_mode("gdas_v1", None)
        elif algo.startswith("darts"):
            network.set_cal_mode("joint", None)
        elif algo == "random":
            network.set_cal_mode("urs", None)
        elif algo != "enas":
            raise ValueError("Invalid algo name : {:}".format(algo))
        network.zero_grad()
        if algo == "darts-v2":
            arch_loss, logits = backward_step_unrolled(
                network,
                criterion,
                base_inputs,
                base_targets,
                w_optimizer,
                arch_inputs,
                arch_targets,
            )
            a_optimizer.step()
        elif algo == "random" or algo == "enas":
            with torch.no_grad():
                _, logits = network(arch_inputs)
                arch_loss = criterion(logits, arch_targets)
        else:
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
    return (
        base_losses.avg,
        base_top1.avg,
        base_top5.avg,
        arch_losses.avg,
        arch_top1.avg,
        arch_top5.avg,
    )


def train_controller(
    xloader, network, criterion, optimizer, prev_baseline, epoch_str, print_freq, logger
):
    # config. (containing some necessary arg)
    #   baseline: The baseline score (i.e. average val_acc) from the previous epoch
    data_time, batch_time = AverageMeter(), AverageMeter()
    (
        GradnormMeter,
        LossMeter,
        ValAccMeter,
        EntropyMeter,
        BaselineMeter,
        RewardMeter,
        xend,
    ) = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        time.time(),
    )

    controller_num_aggregate = 20
    controller_train_steps = 50
    controller_bl_dec = 0.99
    controller_entropy_weight = 0.0001

    network.eval()
    network.controller.train()
    network.controller.zero_grad()
    loader_iter = iter(xloader)
    for step in range(controller_train_steps * controller_num_aggregate):
        try:
            inputs, targets = next(loader_iter)
        except:
            loader_iter = iter(xloader)
            inputs, targets = next(loader_iter)
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - xend)

        log_prob, entropy, sampled_arch = network.controller()
        with torch.no_grad():
            network.set_cal_mode("dynamic", sampled_arch)
            _, logits = network(inputs)
            val_top1, val_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            val_top1 = val_top1.view(-1) / 100
        reward = val_top1 + controller_entropy_weight * entropy
        if prev_baseline is None:
            baseline = val_top1
        else:
            baseline = prev_baseline - (1 - controller_bl_dec) * (
                prev_baseline - reward
            )

        loss = -1 * log_prob * (reward - baseline)

        # account
        RewardMeter.update(reward.item())
        BaselineMeter.update(baseline.item())
        ValAccMeter.update(val_top1.item() * 100)
        LossMeter.update(loss.item())
        EntropyMeter.update(entropy.item())

        # Average gradient over controller_num_aggregate samples
        loss = loss / controller_num_aggregate
        loss.backward(retain_graph=True)

        # measure elapsed time
        batch_time.update(time.time() - xend)
        xend = time.time()
        if (step + 1) % controller_num_aggregate == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                network.controller.parameters(), 5.0
            )
            GradnormMeter.update(grad_norm)
            optimizer.step()
            network.controller.zero_grad()

        if step % print_freq == 0:
            Sstr = (
                "*Train-Controller* "
                + time_string()
                + " [{:}][{:03d}/{:03d}]".format(
                    epoch_str, step, controller_train_steps * controller_num_aggregate
                )
            )
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time
            )
            Wstr = "[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Reward {reward.val:.2f} ({reward.avg:.2f})] Baseline {basel.val:.2f} ({basel.avg:.2f})".format(
                loss=LossMeter,
                top1=ValAccMeter,
                reward=RewardMeter,
                basel=BaselineMeter,
            )
            Estr = "Entropy={:.4f} ({:.4f})".format(EntropyMeter.val, EntropyMeter.avg)
            logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Estr)

    return LossMeter.avg, ValAccMeter.avg, BaselineMeter.avg, RewardMeter.avg


def get_best_arch(xloader, network, n_samples, algo):
    with torch.no_grad():
        network.eval()
        if algo == "random":
            archs, valid_accs = network.return_topK(n_samples, True), []
        elif algo == "setn":
            archs, valid_accs = network.return_topK(n_samples, False), []
        elif algo.startswith("darts") or algo == "gdas" or algo == "gdas_v1":
            arch = network.genotype
            archs, valid_accs = [arch], []
        elif algo == "enas":
            archs, valid_accs = [], []
            for _ in range(n_samples):
                _, _, sampled_arch = network.controller()
                archs.append(sampled_arch)
        else:
            raise ValueError("Invalid algorithm name : {:}".format(algo))
        loader_iter = iter(xloader)
        for i, sampled_arch in enumerate(archs):
            network.set_cal_mode("dynamic", sampled_arch)
            try:
                inputs, targets = next(loader_iter)
            except:
                loader_iter = iter(xloader)
                inputs, targets = next(loader_iter)
            _, logits = network(inputs.cuda(non_blocking=True))
            val_top1, val_top5 = obtain_accuracy(
                logits.cpu().data, targets.data, topk=(1, 5)
            )
            valid_accs.append(val_top1.item())
        best_idx = np.argmax(valid_accs)
        best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
        return best_arch, best_valid_acc


def valid_func(xloader, network, criterion, algo, logger):
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
            _, logits = network(arch_inputs.cuda(non_blocking=True))
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
            C=xargs.channel,
            N=xargs.num_cells,
            max_nodes=xargs.max_nodes,
            num_classes=class_num,
            space=search_space,
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
        api = create(None, "topology", fast_mode=True, verbose=False)
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
        baseline = checkpoint["baseline"]
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
        start_epoch, valid_accuracies, genotypes = (
            0,
            {"best": -1},
            {-1: network.return_topK(1, True)[0]},
        )
        baseline = None

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

        network.set_drop_path(float(epoch + 1) / total_epoch, xargs.drop_path_rate)
        if xargs.algo == "gdas" or xargs.algo == "gdas_v1":
            network.set_tau(
                xargs.tau_max
                - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1)
            )
            logger.log(
                "[RESET tau as : {:} and drop_path as {:}]".format(
                    network.tau, network.drop_path
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
            xargs.algo,
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
        if xargs.algo == "enas":
            ctl_loss, ctl_acc, baseline, ctl_reward = train_controller(
                valid_loader,
                network,
                criterion,
                a_optimizer,
                baseline,
                epoch_str,
                xargs.print_freq,
                logger,
            )
            logger.log(
                "[{:}] controller : loss={:}, acc={:}, baseline={:}, reward={:}".format(
                    epoch_str, ctl_loss, ctl_acc, baseline, ctl_reward
                )
            )

        genotype, temp_accuracy = get_best_arch(
            valid_loader, network, xargs.eval_candidate_num, xargs.algo
        )
        if xargs.algo == "setn" or xargs.algo == "enas":
            network.set_cal_mode("dynamic", genotype)
        elif xargs.algo == "gdas":
            network.set_cal_mode("gdas", None)
        elif xargs.algo == "gdas_v1":
            network.set_cal_mode("gdas_v1", None)
        elif xargs.algo.startswith("darts"):
            network.set_cal_mode("joint", None)
        elif xargs.algo == "random":
            network.set_cal_mode("urs", None)
        else:
            raise ValueError("Invalid algorithm name : {:}".format(xargs.algo))
        logger.log(
            "[{:}] - [get_best_arch] : {:} -> {:}".format(
                epoch_str, genotype, temp_accuracy
            )
        )
        valid_a_loss, valid_a_top1, valid_a_top5 = valid_func(
            valid_loader, network, criterion, xargs.algo, logger
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
                "baseline": baseline,
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
    genotype, temp_accuracy = get_best_arch(
        valid_loader, network, xargs.eval_candidate_num, xargs.algo
    )
    if xargs.algo == "setn" or xargs.algo == "enas":
        network.set_cal_mode("dynamic", genotype)
    elif xargs.algo == "gdas":
        network.set_cal_mode("gdas", None)
    elif xargs.algo == "gdas_v1":
        network.set_cal_mode("gdas_v1", None)
    elif xargs.algo.startswith("darts"):
        network.set_cal_mode("joint", None)
    elif xargs.algo == "random":
        network.set_cal_mode("urs", None)
    else:
        raise ValueError("Invalid algorithm name : {:}".format(xargs.algo))
    search_time.update(time.time() - start_time)

    valid_a_loss, valid_a_top1, valid_a_top5 = valid_func(
        valid_loader, network, criterion, xargs.algo, logger
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
        logger.log("{:}".format(api.query_by_arch(genotype, "200")))
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
        default="tss",
        choices=["tss"],
        help="The search space name.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["darts-v1", "darts-v2", "gdas", "gdas_v1", "setn", "random", "enas"],
        help="The search space name.",
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
    # channels and number-of-cells
    parser.add_argument(
        "--max_nodes", type=int, default=4, help="The maximum number of nodes."
    )
    parser.add_argument(
        "--channel", type=int, default=16, help="The number of channels."
    )
    parser.add_argument(
        "--num_cells", type=int, default=5, help="The number of cells in one stage."
    )
    #
    parser.add_argument(
        "--eval_candidate_num",
        type=int,
        default=100,
        help="The number of selected architectures to evaluate.",
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
    parser.add_argument("--drop_path_rate", type=float, help="The drop path rate.")
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
    if args.overwite_epochs is None:
        args.save_dir = os.path.join(
            "{:}-{:}".format(args.save_dir, args.search_space),
            args.dataset,
            "{:}-affine{:}_BN{:}-{:}".format(
                args.algo, args.affine, args.track_running_stats, args.drop_path_rate
            ),
        )
    else:
        args.save_dir = os.path.join(
            "{:}-{:}".format(args.save_dir, args.search_space),
            args.dataset,
            "{:}-affine{:}_BN{:}-E{:}-{:}".format(
                args.algo,
                args.affine,
                args.track_running_stats,
                args.overwite_epochs,
                args.drop_path_rate,
            ),
        )

    main(args)
