##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
#####################################################################################################
# modified from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py #
#####################################################################################################
# python ./exps/NATS-algos/reinforce.py --dataset cifar10 --search_space tss --learning_rate 0.01
# python ./exps/NATS-algos/reinforce.py --dataset cifar100 --search_space tss --learning_rate 0.01
# python ./exps/NATS-algos/reinforce.py --dataset ImageNet16-120 --search_space tss --learning_rate 0.01
# python ./exps/NATS-algos/reinforce.py --dataset cifar10 --search_space sss --learning_rate 0.01
# python ./exps/NATS-algos/reinforce.py --dataset cifar100 --search_space sss --learning_rate 0.01
# python ./exps/NATS-algos/reinforce.py --dataset ImageNet16-120 --search_space sss --learning_rate 0.01
#####################################################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
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
from nats_bench import create


class PolicyTopology(nn.Module):
    def __init__(self, search_space, max_nodes=4):
        super(PolicyTopology, self).__init__()
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


class PolicySize(nn.Module):
    def __init__(self, search_space):
        super(PolicySize, self).__init__()
        self.candidates = search_space["candidates"]
        self.numbers = search_space["numbers"]
        self.arch_parameters = nn.Parameter(
            1e-3 * torch.randn(self.numbers, len(self.candidates))
        )

    def generate_arch(self, actions):
        channels = [str(self.candidates[i]) for i in actions]
        return ":".join(channels)

    def genotype(self):
        channels = []
        for i in range(self.numbers):
            index = self.arch_parameters[i].argmax().item()
            channels.append(str(self.candidates[index]))
        return ":".join(channels)

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


def main(xargs, api):
    # torch.set_num_threads(4)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    search_space = get_search_spaces(xargs.search_space, "nats-bench")
    if xargs.search_space == "tss":
        policy = PolicyTopology(search_space)
    else:
        policy = PolicySize(search_space)
    optimizer = torch.optim.Adam(policy.parameters(), lr=xargs.learning_rate)
    # optimizer = torch.optim.SGD(policy.parameters(), lr=xargs.learning_rate)
    eps = np.finfo(np.float32).eps.item()
    baseline = ExponentialMovingAverage(xargs.EMA_momentum)
    logger.log("policy    : {:}".format(policy))
    logger.log("optimizer : {:}".format(optimizer))
    logger.log("eps       : {:}".format(eps))

    # nas dataset load
    logger.log("{:} use api : {:}".format(time_string(), api))
    api.reset_time()

    # REINFORCE
    x_start_time = time.time()
    logger.log(
        "Will start searching with time budget of {:} s.".format(xargs.time_budget)
    )
    total_steps, total_costs, trace = 0, [], []
    current_best_index = []
    while len(total_costs) == 0 or total_costs[-1] < xargs.time_budget:
        start_time = time.time()
        log_prob, action = select_action(policy)
        arch = policy.generate_arch(action)
        reward, _, _, current_total_cost = api.simulate_train_eval(
            arch, xargs.dataset, hp="12"
        )
        trace.append((reward, arch))
        total_costs.append(current_total_cost)

        baseline.update(reward)
        # calculate loss
        policy_loss = (-log_prob * (reward - baseline.value())).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        # accumulate time
        total_steps += 1
        logger.log(
            "step [{:3d}] : average-reward={:.3f} : policy_loss={:.4f} : {:}".format(
                total_steps, baseline.value(), policy_loss.item(), policy.genotype()
            )
        )
        # to analyze
        current_best_index.append(
            api.query_index_by_arch(max(trace, key=lambda x: x[0])[1])
        )
    # best_arch = policy.genotype() # first version
    best_arch = max(trace, key=lambda x: x[0])[1]
    logger.log(
        "REINFORCE finish with {:} steps and {:.1f} s (real cost={:.3f}).".format(
            total_steps, total_costs[-1], time.time() - x_start_time
        )
    )
    info = api.query_info_str_by_arch(
        best_arch, "200" if xargs.search_space == "tss" else "90"
    )
    logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()

    return logger.log_dir, current_best_index, total_costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("The REINFORCE Algorithm")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    parser.add_argument(
        "--search_space",
        type=str,
        choices=["tss", "sss"],
        help="Choose the search space.",
    )
    parser.add_argument(
        "--learning_rate", type=float, help="The learning rate for REINFORCE."
    )
    parser.add_argument(
        "--EMA_momentum", type=float, default=0.9, help="The momentum value for EMA."
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--loops_if_rand", type=int, default=500, help="The total runs for evaluation."
    )
    # log
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/search",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--arch_nas_dataset",
        type=str,
        help="The path to load the architecture dataset (tiny-nas-benchmark).",
    )
    parser.add_argument("--print_freq", type=int, help="print frequency (default: 200)")
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()

    api = create(None, args.search_space, fast_mode=True, verbose=False)

    args.save_dir = os.path.join(
        "{:}-{:}".format(args.save_dir, args.search_space),
        "{:}-T{:}".format(args.dataset, args.time_budget),
        "REINFORCE-{:}".format(args.learning_rate),
    )
    print("save-dir : {:}".format(args.save_dir))

    if args.rand_seed < 0:
        save_dir, all_info = None, collections.OrderedDict()
        for i in range(args.loops_if_rand):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, args.loops_if_rand))
            args.rand_seed = random.randint(1, 100000)
            save_dir, all_archs, all_total_times = main(args, api)
            all_info[i] = {"all_archs": all_archs, "all_total_times": all_total_times}
        save_path = save_dir / "results.pth"
        print("save into {:}".format(save_path))
        torch.save(all_info, save_path)
    else:
        main(args, api)
