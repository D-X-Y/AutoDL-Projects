#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/basic-maml.py --env_version v1 --hidden_dim 16 --inner_step 5
# python exps/LFNA/basic-maml.py --env_version v2
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

    def __init__(
        self, network, criterion, epochs, meta_lr, inner_lr=0.01, inner_step=1
    ):
        self.criterion = criterion
        # self.container = container
        self.network = network
        self.meta_optimizer = torch.optim.Adam(
            self.network.parameters(), lr=meta_lr, amsgrad=True
        )
        self.meta_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.meta_optimizer,
            milestones=[
                int(epochs * 0.25),
                int(epochs * 0.5),
                int(epochs * 0.75),
            ],
            gamma=0.3,
        )
        self.inner_lr = inner_lr
        self.inner_step = inner_step
        self._best_info = dict(state_dict=None, iepoch=None, score=None)
        print("There are {:} weights.".format(self.network.get_w_container().numel()))

    def adapt(self, dataset):
        # create a container for the future timestamp
        container = self.network.get_w_container()

        for k in range(0, self.inner_step):
            y_hat = self.network.forward_with_container(dataset.x, container)
            loss = self.criterion(y_hat, dataset.y)
            grads = torch.autograd.grad(loss, container.parameters())
            container = container.additive([-self.inner_lr * grad for grad in grads])
        return container

    def predict(self, x, container=None):
        if container is not None:
            y_hat = self.network.forward_with_container(x, container)
        else:
            y_hat = self.network(x)
        return y_hat

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.meta_optimizer.step()
        self.meta_lr_scheduler.step()

    def zero_grad(self):
        self.meta_optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        self.criterion.load_state_dict(state_dict["criterion"])
        self.network.load_state_dict(state_dict["network"])
        self.meta_optimizer.load_state_dict(state_dict["meta_optimizer"])
        self.meta_lr_scheduler.load_state_dict(state_dict["meta_lr_scheduler"])

    def save_best(self, iepoch, score):
        if self._best_info["score"] is None or self._best_info["score"] < score:
            state_dict = dict(
                criterion=self.criterion.state_dict(),
                network=self.network.state_dict(),
                meta_optimizer=self.meta_optimizer.state_dict(),
                meta_lr_scheduler=self.meta_lr_scheduler.state_dict(),
            )
            self._best_info["state_dict"] = state_dict
            self._best_info["score"] = score
            self._best_info["iepoch"] = iepoch
            is_best = True
        else:
            is_best = False
        return self._best_info, is_best


def main(args):
    logger, env_info, model_kwargs = lfna_setup(args)
    model = get_model(dict(model_type="simple_mlp"), **model_kwargs)

    total_time = env_info["total"]
    for i in range(total_time):
        for xkey in ("timestamp", "x", "y"):
            nkey = "{:}-{:}".format(i, xkey)
            assert nkey in env_info, "{:} no in {:}".format(nkey, list(env_info.keys()))
    train_time_bar = total_time // 2

    criterion = torch.nn.MSELoss()

    maml = MAML(
        model, criterion, args.epochs, args.meta_lr, args.inner_lr, args.inner_step
    )

    # meta-training
    per_epoch_time, start_time = AverageMeter(), time.time()
    # for iepoch in range(args.epochs):
    iepoch = 0
    while iepoch < args.epochs:
        need_time = "Time Left: {:}".format(
            convert_secs2time(per_epoch_time.avg * (args.epochs - iepoch), True)
        )
        logger.log(
            "[{:}] [{:04d}/{:04d}] ".format(time_string(), iepoch, args.epochs)
            + need_time
        )

        maml.zero_grad()
        batch_indexes, meta_losses = [], []
        for ibatch in range(args.meta_batch):
            sampled_timestamp = random.randint(0, train_time_bar)
            batch_indexes.append("{:5d}".format(sampled_timestamp))
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
            future_container = maml.adapt(past_dataset)
            future_y_hat = maml.predict(future_dataset.x, future_container)
            future_loss = maml.criterion(future_y_hat, future_dataset.y)
            meta_losses.append(future_loss)
        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        maml.step()

        logger.log(
            "meta-loss: {:.4f}  batch: {:}".format(
                meta_loss.item(), ",".join(batch_indexes)
            )
        )
        best_info, is_best = maml.save_best(iepoch, -meta_loss.item())
        if is_best:
            save_checkpoint(best_info, logger.path("best"), logger)
            logger.log("Save the best into {:}".format(logger.path("best")))
        if iepoch >= 10 and (
            torch.isnan(meta_loss).item() or meta_loss.item() >= args.fail_thresh
        ):
            xdata = torch.load(logger.path("best"))
            maml.load_state_dict(xdata["state_dict"])
            iepoch = xdata["iepoch"]
            logger.log(
                "The training failed, re-use the previous best epoch [{:}]".format(
                    iepoch
                )
            )
        else:
            iepoch = iepoch + 1
        per_epoch_time.update(time.time() - start_time)
        start_time = time.time()

    w_container_per_epoch = dict()
    for idx in range(1, env_info["total"]):
        past_dataset = TimeData(
            idx - 1,
            env_info["{:}-x".format(idx - 1)],
            env_info["{:}-y".format(idx - 1)],
        )
        current_container = maml.adapt(past_dataset)
        w_container_per_epoch[idx] = current_container.no_grad_clone()
        with torch.no_grad():
            current_x = env_info["{:}-x".format(idx)]
            current_y = env_info["{:}-y".format(idx)]
            current_y_hat = maml.predict(current_x, w_container_per_epoch[idx])
            current_loss = maml.criterion(current_y_hat, current_y)
        logger.log(
            "meta-test: [{:03d}] -> loss={:.4f}".format(idx, current_loss.item())
        )
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
        default="./outputs/lfna-synthetic/use-maml",
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
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=0.05,
        help="The learning rate for the MAML optimizer (default is Adam)",
    )
    parser.add_argument(
        "--fail_thresh",
        type=float,
        default=1000,
        help="The threshold for the failure, which we reuse the previous best model",
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
        default=10,
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
    args.save_dir = "{:}-s{:}-{:}-d{:}".format(
        args.save_dir, args.inner_step, args.env_version, args.hidden_dim
    )
    main(args)