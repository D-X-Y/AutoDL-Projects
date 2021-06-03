#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/GeMOSA/baselines/maml-ft.py --env_version v1 --hidden_dim 16 --inner_step 5 --device cuda
# python exps/GeMOSA/baselines/maml-ft.py --env_version v2 --hidden_dim 16 --inner_step 5 --device cuda
# python exps/GeMOSA/baselines/maml-ft.py --env_version v3 --hidden_dim 32 --inner_step 5 --device cuda
# python exps/GeMOSA/baselines/maml-ft.py --env_version v4 --hidden_dim 32 --inner_step 5 --device cuda
#####################################################
import sys, time, copy, torch, random, argparse
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / ".." / "..").resolve()
print(lib_dir)
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
)
from xautodl.log_utils import time_string
from xautodl.log_utils import AverageMeter, convert_secs2time

from xautodl.procedures.metric_utils import SaveMetric, MSEMetric, Top1AccMetric
from xautodl.datasets.synthetic_core import get_synthetic_env
from xautodl.models.xcore import get_model
from xautodl.xlayers import super_core


class MAML:
    """A LFNA meta-model that uses the MLP as delta-net."""

    def __init__(
        self, network, criterion, epochs, meta_lr, inner_lr=0.01, inner_step=1
    ):
        self.criterion = criterion
        self.network = network
        self.meta_optimizer = torch.optim.Adam(
            self.network.parameters(), lr=meta_lr, amsgrad=True
        )
        self.inner_lr = inner_lr
        self.inner_step = inner_step
        self._best_info = dict(state_dict=None, iepoch=None, score=None)
        print("There are {:} weights.".format(self.network.get_w_container().numel()))

    def adapt(self, x, y):
        # create a container for the future timestamp
        container = self.network.get_w_container()

        for k in range(0, self.inner_step):
            y_hat = self.network.forward_with_container(x, container)
            loss = self.criterion(y_hat, y)
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

    def zero_grad(self):
        self.meta_optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        self.criterion.load_state_dict(state_dict["criterion"])
        self.network.load_state_dict(state_dict["network"])
        self.meta_optimizer.load_state_dict(state_dict["meta_optimizer"])

    def state_dict(self):
        state_dict = dict()
        state_dict["criterion"] = self.criterion.state_dict()
        state_dict["network"] = self.network.state_dict()
        state_dict["meta_optimizer"] = self.meta_optimizer.state_dict()
        return state_dict

    def save_best(self, score):
        success, best_score = self.network.save_best(score)
        return success, best_score

    def load_best(self):
        self.network.load_best()


def main(args):
    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)
    train_env = get_synthetic_env(mode="train", version=args.env_version)
    valid_env = get_synthetic_env(mode="valid", version=args.env_version)
    trainval_env = get_synthetic_env(mode="trainval", version=args.env_version)
    test_env = get_synthetic_env(mode="test", version=args.env_version)
    all_env = get_synthetic_env(mode=None, version=args.env_version)
    logger.log("The training enviornment: {:}".format(train_env))
    logger.log("The validation enviornment: {:}".format(valid_env))
    logger.log("The trainval enviornment: {:}".format(trainval_env))
    logger.log("The total enviornment: {:}".format(all_env))
    logger.log("The test enviornment: {:}".format(test_env))
    model_kwargs = dict(
        config=dict(model_type="norm_mlp"),
        input_dim=all_env.meta_info["input_dim"],
        output_dim=all_env.meta_info["output_dim"],
        hidden_dims=[args.hidden_dim] * 2,
        act_cls="relu",
        norm_cls="layer_norm_1d",
    )

    model = get_model(**model_kwargs)
    model = model.to(args.device)
    if all_env.meta_info["task"] == "regression":
        criterion = torch.nn.MSELoss()
        metric_cls = MSEMetric
    elif all_env.meta_info["task"] == "classification":
        criterion = torch.nn.CrossEntropyLoss()
        metric_cls = Top1AccMetric
    else:
        raise ValueError(
            "This task ({:}) is not supported.".format(all_env.meta_info["task"])
        )

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
            future_idx = random.randint(0, len(trainval_env) - 1)
            future_t, (future_x, future_y) = trainval_env[future_idx]
            # -->>
            seq_times = trainval_env.get_seq_times(future_idx, args.seq_length)
            _, (allxs, allys) = trainval_env.seq_call(seq_times)
            allxs, allys = allxs.view(-1, allxs.shape[-1]), allys.view(-1, 1)
            if trainval_env.meta_info["task"] == "classification":
                allys = allys.view(-1)
            historical_x, historical_y = allxs.to(args.device), allys.to(args.device)
            future_container = maml.adapt(historical_x, historical_y)

            future_x, future_y = future_x.to(args.device), future_y.to(args.device)
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

    def finetune(index):
        seq_times = test_env.get_seq_times(index, args.seq_length)
        _, (allxs, allys) = test_env.seq_call(seq_times)
        allxs, allys = allxs.view(-1, allxs.shape[-1]), allys.view(-1, 1)
        if test_env.meta_info["task"] == "classification":
            allys = allys.view(-1)
        historical_x, historical_y = allxs.to(args.device), allys.to(args.device)
        future_container = maml.adapt(historical_x, historical_y)

        historical_y_hat = maml.predict(historical_x, future_container)
        train_metric = metric_cls(True)
        # model.analyze_weights()
        with torch.no_grad():
            train_metric(historical_y_hat, historical_y)
        train_results = train_metric.get_info()
        return train_results, future_container

    metric = metric_cls(True)
    per_timestamp_time, start_time = AverageMeter(), time.time()
    for idx, (future_time, (future_x, future_y)) in enumerate(test_env):

        need_time = "Time Left: {:}".format(
            convert_secs2time(per_timestamp_time.avg * (len(test_env) - idx), True)
        )
        logger.log(
            "[{:}]".format(time_string())
            + " [{:04d}/{:04d}]".format(idx, len(test_env))
            + " "
            + need_time
        )

        # build optimizer
        train_results, future_container = finetune(idx)

        future_x, future_y = future_x.to(args.device), future_y.to(args.device)
        future_y_hat = maml.predict(future_x, future_container)
        future_loss = criterion(future_y_hat, future_y)
        metric(future_y_hat, future_y)
        log_str = (
            "[{:}]".format(time_string())
            + " [{:04d}/{:04d}]".format(idx, len(test_env))
            + " train-score: {:.5f}, eval-score: {:.5f}".format(
                train_results["score"], metric.get_info()["score"]
            )
        )
        logger.log(log_str)
        logger.log("")
        per_timestamp_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("-" * 200 + "\n")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Use the maml.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/GeMOSA-synthetic/use-maml-ft",
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
        default=0.02,
        help="The learning rate for the MAML optimizer (default is Adam)",
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
        "--seq_length", type=int, default=20, help="The sequence length."
    )
    parser.add_argument(
        "--meta_batch",
        type=int,
        default=256,
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
        "--device",
        type=str,
        default="cpu",
        help="",
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
    args.save_dir = "{:}-s{:}-mlr{:}-d{:}-e{:}-env{:}".format(
        args.save_dir,
        args.inner_step,
        args.meta_lr,
        args.hidden_dim,
        args.epochs,
        args.env_version,
    )
    main(args)
