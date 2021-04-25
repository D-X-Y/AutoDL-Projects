#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.04 #
#####################################################
import os, sys, time, torch
from typing import import Optional, Text, Callable

# modules in AutoDL
from log_utils import AverageMeter
from log_utils import time_string
from .eval_funcs import obtain_accuracy


def basic_train(
    xloader,
    network,
    criterion,
    scheduler,
    optimizer,
    optim_config,
    extra_info,
    print_freq,
    logger,
):
    loss, acc1, acc5 = procedure(
        xloader,
        network,
        criterion,
        scheduler,
        optimizer,
        "train",
        optim_config,
        extra_info,
        print_freq,
        logger,
    )
    return loss, acc1, acc5


def basic_valid(
    xloader, network, criterion, optim_config, extra_info, print_freq, logger
):
    with torch.no_grad():
        loss, acc1, acc5 = procedure(
            xloader,
            network,
            criterion,
            None,
            None,
            "valid",
            None,
            extra_info,
            print_freq,
            logger,
        )
    return loss, acc1, acc5


def procedure(
    xloader,
    network,
    criterion,
    optimizer,
    mode: Text,
    print_freq: int = 100,
    logger_fn: Callable = None
):
    data_time, batch_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    if mode.lower() == "train":
        network.train()
    elif mode.lower() == "valid":
        network.eval()
    else:
        raise ValueError("The mode is not right : {:}".format(mode))

    end = time.time()
    for i, (inputs, targets) in enumerate(xloader):
        # measure data loading time
        data_time.update(time.time() - end)
        # calculate prediction and loss
        targets = targets.cuda(non_blocking=True)

        if mode == "train":
            optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs, targets)

        if mode == "train":
            loss.backward()
            optimizer.step()

        # record
        metrics = 
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or (i + 1) == len(xloader):
            Sstr = (
                " {:5s} ".format(mode.upper())
                + time_string()
                + " [{:}][{:03d}/{:03d}]".format(extra_info, i, len(xloader))
            )
            Lstr = "Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})".format(
                loss=losses, top1=top1, top5=top5
            )
            Istr = "Size={:}".format(list(inputs.size()))
            logger.log(Sstr + " " + Tstr + " " + Lstr + " " + Istr)

    logger.log(
        " **{mode:5s}** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Loss:{loss:.3f}".format(
            mode=mode.upper(),
            top1=top1,
            top5=top5,
            error1=100 - top1.avg,
            error5=100 - top5.avg,
            loss=losses.avg,
        )
    )
    return losses.avg, top1.avg, top5.avg
