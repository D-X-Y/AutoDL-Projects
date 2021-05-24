##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##################################################################
# Regularized Evolution for Image Classifier Architecture Search #
##################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, SearchDataset
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import CellStructure, get_search_spaces
from nas_201_api import NASBench201API as API


class Model(object):
    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return "{:}".format(self.arch)


# This function is to mimic the training and evaluatinig procedure for a single architecture `arch`.
# The time_cost is calculated as the total training time for a few (e.g., 12 epochs) plus the evaluation time for one epoch.
# For use_012_epoch_training = True, the architecture is trained for 12 epochs, with LR being decaded from 0.1 to 0.
#       In this case, the LR schedular is converged.
# For use_012_epoch_training = False, the architecture is planed to be trained for 200 epochs, but we early stop its procedure.
#
def train_and_eval(
    arch, nas_bench, extra_info, dataname="cifar10-valid", use_012_epoch_training=True
):

    if use_012_epoch_training and nas_bench is not None:
        arch_index = nas_bench.query_index_by_arch(arch)
        assert arch_index >= 0, "can not find this arch : {:}".format(arch)
        info = nas_bench.get_more_info(
            arch_index, dataname, iepoch=None, hp="12", is_random=True
        )
        valid_acc, time_cost = (
            info["valid-accuracy"],
            info["train-all-time"] + info["valid-per-time"],
        )
        # _, valid_acc = info.get_metrics('cifar10-valid', 'x-valid' , 25, True) # use the validation accuracy after 25 training epochs
    elif not use_012_epoch_training and nas_bench is not None:
        # Please contact me if you want to use the following logic, because it has some potential issues.
        # Please use `use_012_epoch_training=False` for cifar10 only.
        # It did return values for cifar100 and ImageNet16-120, but it has some potential issues. (Please email me for more details)
        arch_index, nepoch = nas_bench.query_index_by_arch(arch), 25
        assert arch_index >= 0, "can not find this arch : {:}".format(arch)
        xoinfo = nas_bench.get_more_info(
            arch_index, "cifar10-valid", iepoch=None, hp="12"
        )
        xocost = nas_bench.get_cost_info(arch_index, "cifar10-valid", hp="200")
        info = nas_bench.get_more_info(
            arch_index, dataname, nepoch, hp="200", is_random=True
        )  # use the validation accuracy after 25 training epochs, which is used in our ICLR submission (not the camera ready).
        cost = nas_bench.get_cost_info(arch_index, dataname, hp="200")
        # The following codes are used to estimate the time cost.
        # When we build NAS-Bench-201, architectures are trained on different machines and we can not use that time record.
        # When we create checkpoints for converged_LR, we run all experiments on 1080Ti, and thus the time for each architecture can be fairly compared.
        nums = {
            "ImageNet16-120-train": 151700,
            "ImageNet16-120-valid": 3000,
            "cifar10-valid-train": 25000,
            "cifar10-valid-valid": 25000,
            "cifar100-train": 50000,
            "cifar100-valid": 5000,
        }
        estimated_train_cost = (
            xoinfo["train-per-time"]
            / nums["cifar10-valid-train"]
            * nums["{:}-train".format(dataname)]
            / xocost["latency"]
            * cost["latency"]
            * nepoch
        )
        estimated_valid_cost = (
            xoinfo["valid-per-time"]
            / nums["cifar10-valid-valid"]
            * nums["{:}-valid".format(dataname)]
            / xocost["latency"]
            * cost["latency"]
        )
        try:
            valid_acc, time_cost = (
                info["valid-accuracy"],
                estimated_train_cost + estimated_valid_cost,
            )
        except:
            valid_acc, time_cost = (
                info["valtest-accuracy"],
                estimated_train_cost + estimated_valid_cost,
            )
    else:
        # train a model from scratch.
        raise ValueError("NOT IMPLEMENT YET")
    return valid_acc, time_cost


def random_architecture_func(max_nodes, op_names):
    # return a random architecture
    def random_architecture():
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = random.choice(op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    return random_architecture


def mutate_arch_func(op_names):
    """Computes the architecture for a child of the given parent architecture.
    The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
    """

    def mutate_arch_func(parent_arch):
        child_arch = deepcopy(parent_arch)
        node_id = random.randint(0, len(child_arch.nodes) - 1)
        node_info = list(child_arch.nodes[node_id])
        snode_id = random.randint(0, len(node_info) - 1)
        xop = random.choice(op_names)
        while xop == node_info[snode_id][0]:
            xop = random.choice(op_names)
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple(node_info)
        return child_arch

    return mutate_arch_func


def regularized_evolution(
    cycles,
    population_size,
    sample_size,
    time_budget,
    random_arch,
    mutate_arch,
    nas_bench,
    extra_info,
    dataname,
):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      sample_size: the number of individuals that should participate in each tournament.
      time_budget: the upper bound of searching cost

    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    population = collections.deque()
    history, total_time_cost = (
        [],
        0,
    )  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_arch()
        model.accuracy, time_cost = train_and_eval(
            model.arch, nas_bench, extra_info, dataname
        )
        population.append(model)
        history.append(model)
        total_time_cost += time_cost

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    # while len(history) < cycles:
    while total_time_cost < time_budget:
        # Sample randomly chosen models from the current population.
        start_time, sample = time.time(), []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        total_time_cost += time.time() - start_time
        child.accuracy, time_cost = train_and_eval(
            child.arch, nas_bench, extra_info, dataname
        )
        if total_time_cost + time_cost > time_budget:  # return
            return history, total_time_cost
        else:
            total_time_cost += time_cost
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()
    return history, total_time_cost


def main(xargs, nas_bench):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    if xargs.dataset == "cifar10":
        dataname = "cifar10-valid"
    else:
        dataname = xargs.dataset
    if xargs.data_path is not None:
        train_data, valid_data, xshape, class_num = get_datasets(
            xargs.dataset, xargs.data_path, -1
        )
        split_Fpath = "configs/nas-benchmark/cifar-split.txt"
        cifar_split = load_config(split_Fpath, None, None)
        train_split, valid_split = cifar_split.train, cifar_split.valid
        logger.log("Load split file from {:}".format(split_Fpath))
        config_path = "configs/nas-benchmark/algos/R-EA.config"
        config = load_config(
            config_path, {"class_num": class_num, "xshape": xshape}, logger
        )
        # To split data
        train_data_v2 = deepcopy(train_data)
        train_data_v2.transform = valid_data.transform
        valid_data = train_data_v2
        search_data = SearchDataset(xargs.dataset, train_data, train_split, valid_split)
        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
            num_workers=xargs.workers,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
            num_workers=xargs.workers,
            pin_memory=True,
        )
        logger.log(
            "||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(
                xargs.dataset, len(train_loader), len(valid_loader), config.batch_size
            )
        )
        logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))
        extra_info = {
            "config": config,
            "train_loader": train_loader,
            "valid_loader": valid_loader,
        }
    else:
        config_path = "configs/nas-benchmark/algos/R-EA.config"
        config = load_config(config_path, None, logger)
        logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))
        extra_info = {"config": config, "train_loader": None, "valid_loader": None}

    search_space = get_search_spaces("cell", xargs.search_space_name)
    random_arch = random_architecture_func(xargs.max_nodes, search_space)
    mutate_arch = mutate_arch_func(search_space)
    # x =random_arch() ; y = mutate_arch(x)
    x_start_time = time.time()
    logger.log("{:} use nas_bench : {:}".format(time_string(), nas_bench))
    logger.log(
        "-" * 30
        + " start searching with the time budget of {:} s".format(xargs.time_budget)
    )
    history, total_cost = regularized_evolution(
        xargs.ea_cycles,
        xargs.ea_population,
        xargs.ea_sample_size,
        xargs.time_budget,
        random_arch,
        mutate_arch,
        nas_bench if args.ea_fast_by_api else None,
        extra_info,
        dataname,
    )
    logger.log(
        "{:} regularized_evolution finish with history of {:} arch with {:.1f} s (real-cost={:.2f} s).".format(
            time_string(), len(history), total_cost, time.time() - x_start_time
        )
    )
    best_arch = max(history, key=lambda i: i.accuracy)
    best_arch = best_arch.arch
    logger.log("{:} best arch is {:}".format(time_string(), best_arch))

    info = nas_bench.query_by_arch(best_arch, "200")
    if info is None:
        logger.log("Did not find this architecture : {:}.".format(best_arch))
    else:
        logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()
    return logger.log_dir, nas_bench.query_index_by_arch(best_arch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
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
    parser.add_argument("--ea_cycles", type=int, help="The number of cycles in EA.")
    parser.add_argument("--ea_population", type=int, help="The population size in EA.")
    parser.add_argument("--ea_sample_size", type=int, help="The sample size in EA.")
    parser.add_argument(
        "--ea_fast_by_api",
        type=int,
        help="Use our API to speed up the experiments or not.",
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        help="The total time cost budge for searching (in seconds).",
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
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()
    # if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    args.ea_fast_by_api = args.ea_fast_by_api > 0

    if args.arch_nas_dataset is None or not os.path.isfile(args.arch_nas_dataset):
        nas_bench = None
    else:
        print(
            "{:} build NAS-Benchmark-API from {:}".format(
                time_string(), args.arch_nas_dataset
            )
        )
        nas_bench = API(args.arch_nas_dataset)
    if args.rand_seed < 0:
        save_dir, all_indexes, num = None, [], 500
        for i in range(num):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, num))
            args.rand_seed = random.randint(1, 100000)
            save_dir, index = main(args, nas_bench)
            all_indexes.append(index)
        torch.save(all_indexes, save_dir / "results.pth")
    else:
        main(args, nas_bench)
