##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
#####################################################################################################
# modified from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py #
#####################################################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Categorical

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
from R_EA import train_and_eval


class Policy(nn.Module):
    def __init__(self, max_nodes, search_space):
        super(Policy, self).__init__()
        self.max_nodes = max_nodes
        self.search_space = deepcopy(search_space)
        self.edge2index = {}
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                self.edge2index[node_str] = len(self.edge2index)
        self.arch_parameters = nn.Parameter(
            1e-3 * torch.randn(len(self.edge2index), len(search_space))
        )

    def generate_arch(self, actions):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = self.search_space[actions[self.edge2index[node_str]]]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.search_space[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def forward(self):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        return alphas


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

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


def select_action(policy):
    probs = policy()
    m = Categorical(probs)
    action = m.sample()
    # policy.saved_log_probs.append(m.log_prob(action))
    return m.log_prob(action), action.cpu().tolist()


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
        extra_info = {"config": config, "train_loader": None, "valid_loader": None}
        logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    search_space = get_search_spaces("cell", xargs.search_space_name)
    policy = Policy(xargs.max_nodes, search_space)
    optimizer = torch.optim.Adam(policy.parameters(), lr=xargs.learning_rate)
    # optimizer = torch.optim.SGD(policy.parameters(), lr=xargs.learning_rate)
    eps = np.finfo(np.float32).eps.item()
    baseline = ExponentialMovingAverage(xargs.EMA_momentum)
    logger.log("policy    : {:}".format(policy))
    logger.log("optimizer : {:}".format(optimizer))
    logger.log("eps       : {:}".format(eps))

    # nas dataset load
    logger.log("{:} use nas_bench : {:}".format(time_string(), nas_bench))

    # REINFORCE
    # attempts = 0
    x_start_time = time.time()
    logger.log(
        "Will start searching with time budget of {:} s.".format(xargs.time_budget)
    )
    total_steps, total_costs, trace = 0, 0, []
    # for istep in range(xargs.RL_steps):
    while total_costs < xargs.time_budget:
        start_time = time.time()
        log_prob, action = select_action(policy)
        arch = policy.generate_arch(action)
        reward, cost_time = train_and_eval(arch, nas_bench, extra_info, dataname)
        trace.append((reward, arch))
        # accumulate time
        if total_costs + cost_time < xargs.time_budget:
            total_costs += cost_time
        else:
            break

        baseline.update(reward)
        # calculate loss
        policy_loss = (-log_prob * (reward - baseline.value())).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        # accumulate time
        total_costs += time.time() - start_time
        total_steps += 1
        logger.log(
            "step [{:3d}] : average-reward={:.3f} : policy_loss={:.4f} : {:}".format(
                total_steps, baseline.value(), policy_loss.item(), policy.genotype()
            )
        )
        # logger.log('----> {:}'.format(policy.arch_parameters))
        # logger.log('')

    # best_arch = policy.genotype() # first version
    best_arch = max(trace, key=lambda x: x[0])[1]
    logger.log(
        "REINFORCE finish with {:} steps and {:.1f} s (real cost={:.3f}).".format(
            total_steps, total_costs, time.time() - x_start_time
        )
    )
    info = nas_bench.query_by_arch(best_arch, "200")
    if info is None:
        logger.log("Did not find this architecture : {:}.".format(best_arch))
    else:
        logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()
    return logger.log_dir, nas_bench.query_index_by_arch(best_arch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("The REINFORCE Algorithm")
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
        "--learning_rate", type=float, help="The learning rate for REINFORCE."
    )
    # parser.add_argument('--RL_steps',           type=int,   help='The steps for REINFORCE.')
    parser.add_argument(
        "--EMA_momentum", type=float, help="The momentum value for EMA."
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
