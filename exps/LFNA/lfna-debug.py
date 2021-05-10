#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/lfna-debug.py --env_version v1 --hidden_dim 16
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


from lfna_utils import lfna_setup, train_model, TimeData


class LFNAmlp:
    """A LFNA meta-model that uses the MLP as delta-net."""

    def __init__(self, obs_dim, hidden_sizes, act_name, criterion):
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
        self.criterion = criterion

    def adapt(self, model, seq_flatten_w):
        delta_inputs = torch.stack(seq_flatten_w, dim=-1)
        delta = self.delta_net(delta_inputs)
        container = model.get_w_container()
        unflatten_delta = container.unflatten(delta)
        future_container = container.create_container(unflatten_delta)
        return future_container

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.delta_net.parameters(), 1.0)
        self.meta_optimizer.step()

    def zero_grad(self):
        self.meta_optimizer.zero_grad()
        self.delta_net.zero_grad()

    def state_dict(self):
        return dict(
            delta_net=self.delta_net.state_dict(),
            meta_optimizer=self.meta_optimizer.state_dict(),
        )


def main(args):
    logger, env_info, model_kwargs = lfna_setup(args)
    dynamic_env = env_info["dynamic_env"]
    model = get_model(dict(model_type="simple_mlp"), **model_kwargs)

    total_time = env_info["total"]
    for i in range(total_time):
        for xkey in ("timestamp", "x", "y"):
            nkey = "{:}-{:}".format(i, xkey)
            assert nkey in env_info, "{:} no in {:}".format(nkey, list(env_info.keys()))
    train_time_bar = total_time // 2
    network = get_model(dict(model_type="simple_mlp"), **model_kwargs)

    criterion = torch.nn.MSELoss()
    logger.log("There are {:} weights.".format(network.get_w_container().numel()))

    adaptor = LFNAmlp(args.meta_seq, (200, 200), "leaky_relu", criterion)

    # pre-train the model
    init_dataset = TimeData(0, env_info["0-x"], env_info["0-y"])
    init_loss = train_model(network, init_dataset, args.init_lr, args.epochs)
    logger.log("The pre-training loss is {:.4f}".format(init_loss))

    all_past_containers = []
    ground_truth_path = (
        logger.path(None) / ".." / "use-same-timestamp-v1-d16" / "final-ckp.pth"
    )
    ground_truth_data = torch.load(ground_truth_path)
    all_gt_containers = ground_truth_data["w_container_per_epoch"]
    all_gt_flattens = dict()
    for idx, container in all_gt_containers.items():
        all_gt_flattens[idx] = container.no_grad_clone().flatten()

    # LFNA meta-training
    meta_loss_meter = AverageMeter()
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

        meta_losses = []
        for ibatch in range(args.meta_batch):
            future_timestamp = random.randint(args.meta_seq, train_time_bar)
            future_dataset = TimeData(
                future_timestamp,
                env_info["{:}-x".format(future_timestamp)],
                env_info["{:}-y".format(future_timestamp)],
            )
            seq_datasets = []
            for iseq in range(args.meta_seq):
                cur_time = future_timestamp - iseq - 1
                cur_x = env_info["{:}-x".format(cur_time)]
                cur_y = env_info["{:}-y".format(cur_time)]
                seq_datasets.append(TimeData(cur_time, cur_x, cur_y))
            seq_datasets.reverse()
            seq_flatten_w = [
                all_gt_flattens[dataset.timestamp] for dataset in seq_datasets
            ]
            future_container = adaptor.adapt(network, seq_flatten_w)
            """
            future_y_hat = network.forward_with_container(
                future_dataset.x, future_container
            )
            future_loss = adaptor.criterion(future_y_hat, future_dataset.y)
            """
            future_loss = adaptor.criterion(
                future_container.flatten(), all_gt_flattens[future_timestamp]
            )
            # import pdb; pdb.set_trace()
            meta_losses.append(future_loss)
        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        adaptor.step()

        meta_loss_meter.update(meta_loss.item())

        logger.log(
            "meta-loss: {:.4f} ({:.4f}) ".format(
                meta_loss_meter.avg, meta_loss_meter.val
            )
        )
        if iepoch % 200 == 0:
            save_checkpoint(
                {"adaptor": adaptor.state_dict(), "iepoch": iepoch},
                logger.path("model"),
                logger,
            )
        per_epoch_time.update(time.time() - start_time)
        start_time = time.time()

    w_container_per_epoch = dict()
    # import pdb; pdb.set_trace()
    for idx in range(1, env_info["total"]):
        future_time = env_info["{:}-timestamp".format(idx)]
        future_x = env_info["{:}-x".format(idx)]
        future_y = env_info["{:}-y".format(idx)]
        seq_datasets = []
        for iseq in range(1, args.meta_seq + 1):
            cur_time = future_timestamp - iseq - 1
            if cur_time < 0:
                cur_time = 0
            cur_x = env_info["{:}-x".format(cur_time)]
            cur_y = env_info["{:}-y".format(cur_time)]
            seq_datasets.append(TimeData(cur_time, cur_x, cur_y))
        seq_datasets.reverse()
        seq_flatten_w = [all_gt_flattens[dataset.timestamp] for dataset in seq_datasets]
        future_container = adaptor.adapt(network, seq_flatten_w)
        w_container_per_epoch[idx] = future_container.no_grad_clone()
        with torch.no_grad():
            future_y_hat = network.forward_with_container(
                future_x, w_container_per_epoch[idx]
            )
            future_loss = adaptor.criterion(future_y_hat, future_y)
        logger.log("meta-test: [{:03d}] -> loss={:.4f}".format(idx, future_loss.item()))

    save_checkpoint(
        {"w_container_per_epoch": w_container_per_epoch},
        logger.path(None) / "final-ckp.pth",
        logger,
    )

    logger.log("-" * 200 + "\n")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Use the data in the past.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/lfna-synthetic/lfna-debug",
        help="The checkpoint directory.",
    )
    parser.add_argument(
        "--env_version",
        type=str,
        required=True,
        help="The synthetic enviornment version.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        required=True,
        help="The hidden dimension.",
    )
    #####
    parser.add_argument(
        "--init_lr",
        type=float,
        default=0.1,
        help="The initial learning rate for the optimizer (default is Adam)",
    )
    parser.add_argument(
        "--meta_batch",
        type=int,
        default=32,
        help="The batch size for the meta-model",
    )
    parser.add_argument(
        "--meta_seq",
        type=int,
        default=10,
        help="The length of the sequence for meta-model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="The total number of epochs.",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "The save dir argument can not be None"
    args.save_dir = "{:}-{:}-d{:}".format(
        args.save_dir, args.env_version, args.hidden_dim
    )
    main(args)
