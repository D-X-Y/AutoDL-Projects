#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/lfna-v0.py
#####################################################
import sys, time, copy, torch, random, argparse
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from procedures import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint
from log_utils import time_string
from log_utils import AverageMeter, convert_secs2time

from utils import split_str2indexes

from procedures.advanced_main import basic_train_fn, basic_eval_fn
from procedures.metric_utils import SaveMetric, MSEMetric, ComposeMetric
from datasets.synthetic_core import get_synthetic_env
from models.xcore import get_model
from xlayers import super_core


class LFNAmlp:
    """A LFNA meta-model that uses the MLP as delta-net."""

    def __init__(self, obs_dim, hidden_sizes, act_name):
        self.delta_net = super_core.SuperSequential(
            super_core.SuperLinear(obs_dim, hidden_sizes[0]),
            super_core.super_name2activation[act_name](),
            super_core.SuperLinear(hidden_sizes[0], hidden_sizes[1]),
            super_core.super_name2activation[act_name](),
            super_core.SuperLinear(hidden_sizes[1], 1),
        )
        self.meta_optimizer = torch.optim.Adam(
            self.delta_net.parameters(), lr=0.01, amsgrad=True
        )

    def adapt(self, model, criterion, w_container, seq_datasets):
        w_container.requires_grad_(True)
        containers = [w_container]
        for idx, dataset in enumerate(seq_datasets):
            x, y = dataset.x, dataset.y
            y_hat = model.forward_with_container(x, containers[-1])
            loss = criterion(y_hat, y)
            gradients = torch.autograd.grad(loss, containers[-1].tensors)
            with torch.no_grad():
                flatten_w = containers[-1].flatten().view(-1, 1)
                flatten_g = containers[-1].flatten(gradients).view(-1, 1)
                input_statistics = torch.tensor([x.mean(), x.std()]).view(1, 2)
                input_statistics = input_statistics.expand(flatten_w.numel(), -1)
            delta_inputs = torch.cat((flatten_w, flatten_g, input_statistics), dim=-1)
            delta = self.delta_net(delta_inputs).view(-1)
            delta = torch.clamp(delta, -0.5, 0.5)
            unflatten_delta = containers[-1].unflatten(delta)
            future_container = containers[-1].no_grad_clone().additive(unflatten_delta)
            # future_container = containers[-1].additive(unflatten_delta)
            containers.append(future_container)
        # containers = containers[1:]
        meta_loss = []
        temp_containers = []
        for idx, dataset in enumerate(seq_datasets):
            if idx == 0:
                continue
            current_container = containers[idx]
            y_hat = model.forward_with_container(dataset.x, current_container)
            loss = criterion(y_hat, dataset.y)
            meta_loss.append(loss)
            temp_containers.append((dataset.timestamp, current_container, -loss.item()))
        meta_loss = sum(meta_loss)
        w_container.requires_grad_(False)
        # meta_loss.backward()
        # self.meta_optimizer.step()
        return meta_loss, temp_containers

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.delta_net.parameters(), 1.0)
        self.meta_optimizer.step()

    def zero_grad(self):
        self.meta_optimizer.zero_grad()
        self.delta_net.zero_grad()


class TimeData:
    def __init__(self, timestamp, xs, ys):
        self._timestamp = timestamp
        self._xs = xs
        self._ys = ys

    @property
    def x(self):
        return self._xs

    @property
    def y(self):
        return self._ys

    @property
    def timestamp(self):
        return self._timestamp


class Population:
    """A population used to maintain models at different timestamps."""

    def __init__(self):
        self._time2model = dict()
        self._time2score = dict()  # higher is better

    def append(self, timestamp, model, score):
        if timestamp in self._time2model:
            if self._time2score[timestamp] > score:
                return
        self._time2model[timestamp] = model.no_grad_clone()
        self._time2score[timestamp] = score

    def query(self, timestamp):
        closet_timestamp = None
        for xtime, model in self._time2model.items():
            if closet_timestamp is None or (
                xtime < timestamp and timestamp - closet_timestamp >= timestamp - xtime
            ):
                closet_timestamp = xtime
        return self._time2model[closet_timestamp], closet_timestamp

    def debug_info(self, timestamps):
        xstrs = []
        for timestamp in timestamps:
            if timestamp in self._time2score:
                xstrs.append(
                    "{:04d}: {:.4f}".format(timestamp, self._time2score[timestamp])
                )
        return ", ".join(xstrs)


def main(args):
    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)

    cache_path = (logger.path(None) / ".." / "env-info.pth").resolve()
    if cache_path.exists():
        env_info = torch.load(cache_path)
    else:
        env_info = dict()
        dynamic_env = get_synthetic_env()
        env_info["total"] = len(dynamic_env)
        for idx, (timestamp, (_allx, _ally)) in enumerate(tqdm(dynamic_env)):
            env_info["{:}-timestamp".format(idx)] = timestamp
            env_info["{:}-x".format(idx)] = _allx
            env_info["{:}-y".format(idx)] = _ally
        env_info["dynamic_env"] = dynamic_env
        torch.save(env_info, cache_path)

    total_time = env_info["total"]
    for i in range(total_time):
        for xkey in ("timestamp", "x", "y"):
            nkey = "{:}-{:}".format(i, xkey)
            assert nkey in env_info, "{:} no in {:}".format(nkey, list(env_info.keys()))
    train_time_bar = total_time // 2
    base_model = get_model(
        dict(model_type="simple_mlp"),
        act_cls="leaky_relu",
        norm_cls="identity",
        input_dim=1,
        output_dim=1,
    )

    w_container = base_model.get_w_container()
    criterion = torch.nn.MSELoss()
    print("There are {:} weights.".format(w_container.numel()))

    adaptor = LFNAmlp(4, (50, 20), "leaky_relu")

    pool = Population()
    pool.append(0, w_container, -100)

    # LFNA meta-training
    per_epoch_time, start_time = AverageMeter(), time.time()
    for iepoch in range(args.epochs):

        need_time = "Time Left: {:}".format(
            convert_secs2time(per_epoch_time.avg * (args.epochs - iepoch), True)
        )
        logger.log(
            "[{:}] [{:04d}/{:04d}] ".format(time_string(), iepoch, args.epochs)
            + need_time
        )

        adaptor.zero_grad()

        debug_timestamp = set()
        all_meta_losses = []
        for ibatch in range(args.meta_batch):
            sampled_timestamp = random.randint(0, train_time_bar)
            query_w_container, query_timestamp = pool.query(sampled_timestamp)
            # def adapt(self, model, w_container, xs, ys):
            seq_datasets = []
            # xs, ys = [], []
            for it in range(sampled_timestamp, sampled_timestamp + args.max_seq):
                xs = env_info["{:}-x".format(it)]
                ys = env_info["{:}-y".format(it)]
                seq_datasets.append(TimeData(it, xs, ys))
            temp_meta_loss, temp_containers = adaptor.adapt(
                base_model, criterion, query_w_container, seq_datasets
            )
            all_meta_losses.append(temp_meta_loss)
            for temp_time, temp_container, temp_score in temp_containers:
                pool.append(temp_time, temp_container, temp_score)
                debug_timestamp.add(temp_time)
        meta_loss = torch.stack(all_meta_losses).mean()
        meta_loss.backward()
        adaptor.step()

        debug_str = pool.debug_info(debug_timestamp)
        logger.log("meta-loss: {:.4f}".format(meta_loss.item()))

        per_epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("-" * 200 + "\n")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Use the data in the past.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/lfna-synthetic/lfna-v1",
        help="The checkpoint directory.",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        default=0.1,
        help="The initial learning rate for the optimizer (default is Adam)",
    )
    parser.add_argument(
        "--meta_batch",
        type=int,
        default=5,
        help="The batch size for the meta-model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="The total number of epochs.",
    )
    parser.add_argument(
        "--max_seq",
        type=int,
        default=5,
        help="The maximum length of the sequence.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="The number of data loading workers (default: 4)",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "The save dir argument can not be None"
    main(args)
