##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.07                          #
##############################################################################
# python ./exps/NATS-Bench/show-dataset.py                                   #
##############################################################################
import os, sys, time, torch, random, argparse
from typing import List, Text, Dict, Any
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy

from xautodl.config_utils import dict2config, load_config
from xautodl.datasets import get_datasets
from nats_bench import create


def show_imagenet_16_120(dataset_dir=None):
    if dataset_dir is None:
        torch_home_dir = (
            os.environ["TORCH_HOME"]
            if "TORCH_HOME" in os.environ
            else os.path.join(os.environ["HOME"], ".torch")
        )
        dataset_dir = os.path.join(torch_home_dir, "cifar.python", "ImageNet16")
    train_data, valid_data, xshape, class_num = get_datasets(
        "ImageNet16-120", dataset_dir, -1
    )
    split_info = load_config(
        "configs/nas-benchmark/ImageNet16-120-split.txt", None, None
    )
    print("=" * 10 + " ImageNet-16-120 " + "=" * 10)
    print("Training Data: {:}".format(train_data))
    print("Evaluation Data: {:}".format(valid_data))
    print("Hold-out training: {:} images.".format(len(split_info.train)))
    print("Hold-out valid   : {:} images.".format(len(split_info.valid)))


if __name__ == "__main__":
    # show_imagenet_16_120()
    api_nats_tss = create(None, "tss", fast_mode=True, verbose=True)

    valid_acc_12e = []
    test_acc_12e = []
    test_acc_200e = []
    for index in range(10000):
        info = api_nats_tss.get_more_info(index, "ImageNet16-120", hp="12")
        valid_acc_12e.append(
            info["valid-accuracy"]
        )  # the validation accuracy after training the model by 12 epochs
        test_acc_12e.append(
            info["test-accuracy"]
        )  # the test accuracy after training the model by 12 epochs
        info = api_nats_tss.get_more_info(index, "ImageNet16-120", hp="200")
        test_acc_200e.append(
            info["test-accuracy"]
        )  # the test accuracy after training the model by 200 epochs (which I reported in the paper)
