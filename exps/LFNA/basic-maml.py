#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/basic-maml.py --env_version v1   #
# python exps/LFNA/basic-maml.py --env_version v2   #
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

from lfna_utils import lfna_setup, TimeData


class MAML:
    """A LFNA meta-model that uses the MLP as delta-net."""

    def __init__(self, container, criterion, meta_lr, inner_lr=0.01, inner_step=1):
        self.criterion = criterion
        self.container = container
        self.meta_optimizer = torch.optim.Adam(
            self.container.parameters(), lr=meta_lr, amsgrad=True
        )
        self.inner_lr = inner_lr
        self.inner_step = inner_step

    def adapt(self, model, dataset):
        # create a container for the future timestamp
        y_hat = model.forward_with_container(dataset.x, self.container)
        loss = self.criterion(y_hat, dataset.y)
        grads = torch.autograd.grad(loss, self.container.parameters())

        fast_container = self.container.additive(
            [-self.inner_lr * grad for grad in grads]
        )
        import pdb

        pdb.set_trace()
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


def main(args):
    logger, env_info = lfna_setup(args)

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

    maml = MAML(w_container, criterion, args.meta_lr, args.inner_lr, args.inner_step)

    # meta-training
    per_epoch_time, start_time = AverageMeter(), time.time()
    for iepoch in range(args.epochs):

        need_time = "Time Left: {:}".format(
            convert_secs2time(per_epoch_time.avg * (args.epochs - iepoch), True)
        )
        logger.log(
            "[{:}] [{:04d}/{:04d}] ".format(time_string(), iepoch, args.epochs)
            + need_time
        )

        maml.zero_grad()

        all_meta_losses = []
        for ibatch in range(args.meta_batch):
            sampled_timestamp = random.randint(0, train_time_bar)
            past_dataset = TimeData(
                sampled_timestamp,
                env_info["{:}-x".format(sampled_timestamp)],
                env_info["{:}-y".format(sampled_timestamp)],
            )
            future_dataset = TimeData(
                sampled_timestamp + 1,
                env_info["{:}-x".format(sampled_timestamp + 1)],
                env_info["{:}-y".format(sampled_timestamp + 1)],
            )
            maml.adapt(base_model, past_dataset)
            import pdb

            pdb.set_trace()

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
        default="./outputs/lfna-synthetic/maml",
        help="The checkpoint directory.",
    )
    parser.add_argument(
        "--env_version",
        type=str,
        required=True,
        help="The synthetic enviornment version.",
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=0.01,
        help="The learning rate for the MAML optimizer (default is Adam)",
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=0.01,
        help="The learning rate for the inner optimization",
    )
    parser.add_argument(
        "--inner_step", type=int, default=1, help="The inner loop steps for MAML."
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
