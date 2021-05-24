##############################################################################
# NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.07                          #
##############################################################################
# python ./exps/NATS-Bench/Analyze-time.py                                   #
##############################################################################
import os, sys, time, tqdm, argparse
from pathlib import Path

from xautodl.config_utils import dict2config, load_config
from xautodl.datasets import get_datasets
from nats_bench import create


def show_time(api, epoch=12):
    print("Show the time for {:} with {:}-epoch-training".format(api, epoch))
    all_cifar10_time, all_cifar100_time, all_imagenet_time = 0, 0, 0
    for index in tqdm.tqdm(range(len(api))):
        info = api.get_more_info(index, "ImageNet16-120", hp=epoch)
        imagenet_time = info["train-all-time"]
        info = api.get_more_info(index, "cifar10-valid", hp=epoch)
        cifar10_time = info["train-all-time"]
        info = api.get_more_info(index, "cifar100", hp=epoch)
        cifar100_time = info["train-all-time"]
        # accumulate the time
        all_cifar10_time += cifar10_time
        all_cifar100_time += cifar100_time
        all_imagenet_time += imagenet_time
    print(
        "The total training time for CIFAR-10        (held-out train set) is {:} seconds".format(
            all_cifar10_time
        )
    )
    print(
        "The total training time for CIFAR-100       (held-out train set) is {:} seconds, {:.2f} times longer than that on CIFAR-10".format(
            all_cifar100_time, all_cifar100_time / all_cifar10_time
        )
    )
    print(
        "The total training time for ImageNet-16-120 (held-out train set) is {:} seconds, {:.2f} times longer than that on CIFAR-10".format(
            all_imagenet_time, all_imagenet_time / all_cifar10_time
        )
    )


if __name__ == "__main__":

    api_nats_tss = create(None, "tss", fast_mode=True, verbose=False)
    show_time(api_nats_tss, 12)

    api_nats_sss = create(None, "sss", fast_mode=True, verbose=False)
    show_time(api_nats_sss, 12)
