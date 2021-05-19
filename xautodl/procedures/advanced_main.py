#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.04 #
#####################################################
# To be finished.
#
import os, sys, time, torch
from typing import Optional, Text, Callable

# modules in AutoDL
from xautodl.log_utils import AverageMeter, time_string
from .eval_funcs import obtain_accuracy


def get_device(tensors):
    if isinstance(tensors, (list, tuple)):
        return get_device(tensors[0])
    elif isinstance(tensors, dict):
        for key, value in tensors.items():
            return get_device(value)
    else:
        return tensors.device


def basic_train_fn(
    xloader,
    network,
    criterion,
    optimizer,
    metric,
    logger,
):
    results = procedure(
        xloader,
        network,
        criterion,
        optimizer,
        metric,
        "train",
        logger,
    )
    return results


def basic_eval_fn(xloader, network, metric, logger):
    with torch.no_grad():
        results = procedure(
            xloader,
            network,
            None,
            None,
            metric,
            "valid",
            logger,
        )
    return results


def procedure(
    xloader,
    network,
    criterion,
    optimizer,
    metric,
    mode: Text,
    logger_fn: Callable = None,
):
    data_time, batch_time = AverageMeter(), AverageMeter()
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

        if mode == "train":
            optimizer.zero_grad()

        outputs = network(inputs)
        targets = targets.to(get_device(outputs))

        if mode == "train":
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # record
        with torch.no_grad():
            results = metric(outputs, targets)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return metric.get_info()
