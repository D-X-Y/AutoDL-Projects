##############################################################################
# NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08                          #
##############################################################################
# Usage: python exps/NATS-Bench/test-nats-api.py                             #
##############################################################################
import os, gc, sys, time, torch, argparse
import numpy as np
from typing import List, Text, Dict, Any
from shutil import copyfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import matplotlib
import seaborn as sns

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from xautodl.config_utils import dict2config, load_config
from xautodl.log_utils import time_string
from xautodl.models import get_cell_based_tiny_net, CellStructure
from nats_bench import create


def test_api(api, sss_or_tss=True):
    print("{:} start testing the api : {:}".format(time_string(), api))
    api.clear_params(12)
    api.reload(index=12)

    # Query the informations of 1113-th architecture
    info_strs = api.query_info_str_by_arch(1113)
    print(info_strs)
    info = api.query_by_index(113)
    print("{:}\n".format(info))
    info = api.query_by_index(113, "cifar100")
    print("{:}\n".format(info))

    info = api.query_meta_info_by_index(115, "90" if sss_or_tss else "200")
    print("{:}\n".format(info))

    for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        for xset in ["train", "test", "valid"]:
            best_index, highest_accuracy = api.find_best(dataset, xset)
        print("")
    params = api.get_net_param(12, "cifar10", None)

    # Obtain the config and create the network
    config = api.get_net_config(12, "cifar10")
    print("{:}\n".format(config))
    network = get_cell_based_tiny_net(config)
    network.load_state_dict(next(iter(params.values())))

    # Obtain the cost information
    info = api.get_cost_info(12, "cifar10")
    print("{:}\n".format(info))
    info = api.get_latency(12, "cifar10")
    print("{:}\n".format(info))
    for index in [13, 15, 19, 200]:
        info = api.get_latency(index, "cifar10")

    # Count the number of architectures
    info = api.statistics("cifar100", "12")
    print("{:} statistics results : {:}\n".format(time_string(), info))

    # Show the information of the 123-th architecture
    api.show(123)

    # Obtain both cost and performance information
    info = api.get_more_info(1234, "cifar10")
    print("{:}\n".format(info))
    print("{:} finish testing the api : {:}".format(time_string(), api))

    if not sss_or_tss:
        arch_str = "|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|"
        matrix = api.str2matrix(arch_str)
        print("Compute the adjacency matrix of {:}".format(arch_str))
        print(matrix)
    info = api.simulate_train_eval(123, "cifar10")
    print("simulate_train_eval : {:}\n\n".format(info))


if __name__ == "__main__":

    # api201 = create('./output/NATS-Bench-topology/process-FULL', 'topology', fast_mode=True, verbose=True)
    for fast_mode in [True, False]:
        for verbose in [True, False]:
            api_nats_tss = create(None, "tss", fast_mode=fast_mode, verbose=True)
            print(
                "{:} create with fast_mode={:} and verbose={:}".format(
                    time_string(), fast_mode, verbose
                )
            )
            test_api(api_nats_tss, False)
            del api_nats_tss
            gc.collect()

    for fast_mode in [True, False]:
        for verbose in [True, False]:
            print(
                "{:} create with fast_mode={:} and verbose={:}".format(
                    time_string(), fast_mode, verbose
                )
            )
            api_nats_sss = create(None, "size", fast_mode=fast_mode, verbose=True)
            print("{:} --->>> {:}".format(time_string(), api_nats_sss))
            test_api(api_nats_sss, True)
            del api_nats_sss
            gc.collect()
