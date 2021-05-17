#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/basic-maml.py --env_version v1 --inner_step 5
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
from datasets.synthetic_core import get_synthetic_env, EnvSampler
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
                int(epochs * 0.8),
                int(epochs * 0.9),
            ],
            gamma=0.1,
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

    def state_dict(self):
        state_dict = dict()
        state_dict["criterion"] = self.criterion.state_dict()
        state_dict["network"] = self.network.state_dict()
        state_dict["meta_optimizer"] = self.meta_optimizer.state_dict()
        state_dict["meta_lr_scheduler"] = self.meta_lr_scheduler.state_dict()
        return state_dict

    def save_best(self, score):
        success, best_score = self.network.save_best(score)
        return success, best_score

    def load_best(self):
        self.network.load_best()


def main(args):
    logger, env_info, model_kwargs = lfna_setup(args)
    model = get_model(**model_kwargs)

    dynamic_env = get_synthetic_env(mode="train", version=args.env_version)

    criterion = torch.nn.MSELoss()

    maml = MAML(
        model, criterion, args.epochs, args.meta_lr, args.inner_lr, args.inner_step
    )

    # meta-training
    last_success_epoch = 0
    per_epoch_time, start_time = AverageMeter(), time.time()
    for iepoch in range(args.epochs):
        need_time = "Time Left: {:}".format(
            convert_secs2time(per_epoch_time.avg * (args.epochs - iepoch), True)
        )
        head_str = (
            "[{:}] [{:04d}/{:04d}] ".format(time_string(), iepoch, args.epochs)
            + need_time
        )

        maml.zero_grad()
        meta_losses = []
        for ibatch in range(args.meta_batch):
            future_timestamp = dynamic_env.random_timestamp()
            _, (future_x, future_y) = dynamic_env(future_timestamp)
            past_timestamp = (
                future_timestamp - args.prev_time * dynamic_env.timestamp_interval
            )
            _, (past_x, past_y) = dynamic_env(past_timestamp)

            future_container = maml.adapt(TimeData(past_timestamp, past_x, past_y))
            future_y_hat = maml.predict(future_x, future_container)
            future_loss = maml.criterion(future_y_hat, future_y)
            meta_losses.append(future_loss)
        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        maml.step()

        logger.log(head_str + " meta-loss: {:.4f}".format(meta_loss.item()))
        success, best_score = maml.save_best(-meta_loss.item())
        if success:
            logger.log("Achieve the best with best_score = {:.3f}".format(best_score))
            save_checkpoint(maml.state_dict(), logger.path("model"), logger)
            last_success_epoch = iepoch
        if iepoch - last_success_epoch >= args.early_stop_thresh:
            logger.log("Early stop at {:}".format(iepoch))
            break

        per_epoch_time.update(time.time() - start_time)
        start_time = time.time()

    # meta-test
    maml.load_best()
    eval_env = env_info["dynamic_env"]
    assert eval_env.timestamp_interval == dynamic_env.timestamp_interval
    w_container_per_epoch = dict()
    for idx in range(args.prev_time, len(eval_env)):
        future_timestamp, (future_x, future_y) = eval_env[idx]
        past_timestamp = (
            future_timestamp.item() - args.prev_time * eval_env.timestamp_interval
        )
        _, (past_x, past_y) = eval_env(past_timestamp)
        future_container = maml.adapt(TimeData(past_timestamp, past_x, past_y))
        w_container_per_epoch[idx] = future_container.no_grad_clone()
        with torch.no_grad():
            future_y_hat = maml.predict(future_x, w_container_per_epoch[idx])
            future_loss = maml.criterion(future_y_hat, future_y)
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
        default=16,
        help="The hidden dimension.",
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=0.01,
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
        default=0.005,
        help="The learning rate for the inner optimization",
    )
    parser.add_argument(
        "--inner_step", type=int, default=1, help="The inner loop steps for MAML."
    )
    parser.add_argument(
        "--prev_time",
        type=int,
        default=5,
        help="The gap between prev_time and current_timestamp",
    )
    parser.add_argument(
        "--meta_batch",
        type=int,
        default=64,
        help="The batch size for the meta-model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="The total number of epochs.",
    )
    parser.add_argument(
        "--early_stop_thresh",
        type=int,
        default=50,
        help="The maximum epochs for early stop.",
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
    args.save_dir = "{:}-s{:}-mlr{:}-d{:}-prev{:}-e{:}-env{:}".format(
        args.save_dir,
        args.inner_step,
        args.meta_lr,
        args.hidden_dim,
        args.prev_time,
        args.epochs,
        args.env_version,
    )
    main(args)
