#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/basic-his.py --srange 1-999 --env_version v1 --hidden_dim 16
#####################################################
import sys, time, copy, torch, random, argparse
from tqdm import tqdm
from copy import deepcopy

from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
)
from xautodl.log_utils import time_string
from xautodl.log_utils import AverageMeter, convert_secs2time

from xautodl.utils import split_str2indexes

from xautodl.procedures.advanced_main import basic_train_fn, basic_eval_fn
from xautodl.procedures.metric_utils import SaveMetric, MSEMetric, ComposeMetric
from xautodl.datasets.synthetic_core import get_synthetic_env
from xautodl.models.xcore import get_model


from lfna_utils import lfna_setup


def subsample(historical_x, historical_y, maxn=10000):
    total = historical_x.size(0)
    if total <= maxn:
        return historical_x, historical_y
    else:
        indexes = torch.randint(low=0, high=total, size=[maxn])
        return historical_x[indexes], historical_y[indexes]


def main(args):
    logger, env_info, model_kwargs = lfna_setup(args)

    # check indexes to be evaluated
    to_evaluate_indexes = split_str2indexes(args.srange, env_info["total"], None)
    logger.log(
        "Evaluate {:}, which has {:} timestamps in total.".format(
            args.srange, len(to_evaluate_indexes)
        )
    )

    w_container_per_epoch = dict()

    per_timestamp_time, start_time = AverageMeter(), time.time()
    for i, idx in enumerate(to_evaluate_indexes):

        need_time = "Time Left: {:}".format(
            convert_secs2time(
                per_timestamp_time.avg * (len(to_evaluate_indexes) - i), True
            )
        )
        logger.log(
            "[{:}]".format(time_string())
            + " [{:04d}/{:04d}][{:04d}]".format(i, len(to_evaluate_indexes), idx)
            + " "
            + need_time
        )
        # train the same data
        assert idx != 0
        historical_x, historical_y = [], []
        for past_i in range(idx):
            historical_x.append(env_info["{:}-x".format(past_i)])
            historical_y.append(env_info["{:}-y".format(past_i)])
        historical_x, historical_y = torch.cat(historical_x), torch.cat(historical_y)
        historical_x, historical_y = subsample(historical_x, historical_y)
        # build model
        model = get_model(dict(model_type="simple_mlp"), **model_kwargs)
        # build optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, amsgrad=True)
        criterion = torch.nn.MSELoss()
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(args.epochs * 0.25),
                int(args.epochs * 0.5),
                int(args.epochs * 0.75),
            ],
            gamma=0.3,
        )
        train_metric = MSEMetric()
        best_loss, best_param = None, None
        for _iepoch in range(args.epochs):
            preds = model(historical_x)
            optimizer.zero_grad()
            loss = criterion(preds, historical_y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # save best
            if best_loss is None or best_loss > loss.item():
                best_loss = loss.item()
                best_param = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_param)
        with torch.no_grad():
            train_metric(preds, historical_y)
        train_results = train_metric.get_info()

        metric = ComposeMetric(MSEMetric(), SaveMetric())
        eval_dataset = torch.utils.data.TensorDataset(
            env_info["{:}-x".format(idx)], env_info["{:}-y".format(idx)]
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        results = basic_eval_fn(eval_loader, model, metric, logger)
        log_str = (
            "[{:}]".format(time_string())
            + " [{:04d}/{:04d}]".format(idx, env_info["total"])
            + " train-mse: {:.5f}, eval-mse: {:.5f}".format(
                train_results["mse"], results["mse"]
            )
        )
        logger.log(log_str)

        save_path = logger.path(None) / "{:04d}-{:04d}.pth".format(
            idx, env_info["total"]
        )
        w_container_per_epoch[idx] = model.get_w_container().no_grad_clone()
        save_checkpoint(
            {
                "model_state_dict": model.state_dict(),
                "model": model,
                "index": idx,
                "timestamp": env_info["{:}-timestamp".format(idx)],
            },
            save_path,
            logger,
        )
        logger.log("")
        per_timestamp_time.update(time.time() - start_time)
        start_time = time.time()

    save_checkpoint(
        {"w_container_per_epoch": w_container_per_epoch},
        logger.path(None) / "final-ckp.pth",
        logger,
    )
    logger.log("-" * 200 + "\n")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Use all the past data to train.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/lfna-synthetic/use-all-past-data",
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
        "--init_lr",
        type=float,
        default=0.1,
        help="The initial learning rate for the optimizer (default is Adam)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="The batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="The total number of epochs.",
    )
    parser.add_argument(
        "--srange", type=str, required=True, help="The range of models to be evaluated"
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
    args.save_dir = "{:}-{:}-d{:}".format(
        args.save_dir, args.env_version, args.hidden_dim
    )
    main(args)
