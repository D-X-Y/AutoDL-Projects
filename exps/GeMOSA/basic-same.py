#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/GeMOSA/basic-same.py --env_version v1 --hidden_dim 16 --epochs 500 --init_lr 0.1 --device cuda
# python exps/GeMOSA/basic-same.py --env_version v2 --hidden_dim 16 --epochs 500 --init_lr 0.1 --device cuda
# python exps/GeMOSA/basic-same.py --env_version v3 --hidden_dim 32 --epochs 1000 --init_lr 0.05 --device cuda
# python exps/GeMOSA/basic-same.py --env_version v4 --hidden_dim 32 --epochs 1000 --init_lr 0.05 --device cuda
#####################################################
import sys, time, copy, torch, random, argparse
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "..").resolve()
print("LIB-DIR: {:}".format(lib_dir))
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

from xautodl.utils import split_str2indexes

from xautodl.procedures.metric_utils import (
    SaveMetric,
    MSEMetric,
    Top1AccMetric,
    ComposeMetric,
)
from xautodl.datasets.synthetic_core import get_synthetic_env
from xautodl.models.xcore import get_model


def subsample(historical_x, historical_y, maxn=10000):
    total = historical_x.size(0)
    if total <= maxn:
        return historical_x, historical_y
    else:
        indexes = torch.randint(low=0, high=total, size=[maxn])
        return historical_x[indexes], historical_y[indexes]


def main(args):
    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)
    env = get_synthetic_env(mode=None, version=args.env_version)
    model_kwargs = dict(
        config=dict(model_type="norm_mlp"),
        input_dim=env.meta_info["input_dim"],
        output_dim=env.meta_info["output_dim"],
        hidden_dims=[args.hidden_dim] * 2,
        act_cls="relu",
        norm_cls="layer_norm_1d",
    )
    logger.log("The total enviornment: {:}".format(env))
    w_containers = dict()

    if env.meta_info["task"] == "regression":
        criterion = torch.nn.MSELoss()
        metric_cls = MSEMetric
    elif env.meta_info["task"] == "classification":
        criterion = torch.nn.CrossEntropyLoss()
        metric_cls = Top1AccMetric
    else:
        raise ValueError(
            "This task ({:}) is not supported.".format(all_env.meta_info["task"])
        )

    per_timestamp_time, start_time = AverageMeter(), time.time()
    for idx, (future_time, (future_x, future_y)) in enumerate(env):

        need_time = "Time Left: {:}".format(
            convert_secs2time(per_timestamp_time.avg * (len(env) - idx), True)
        )
        logger.log(
            "[{:}]".format(time_string())
            + " [{:04d}/{:04d}]".format(idx, len(env))
            + " "
            + need_time
        )
        # train the same data
        historical_x = future_x.to(args.device)
        historical_y = future_y.to(args.device)
        # build model
        model = get_model(**model_kwargs)
        model = model.to(args.device)
        if idx == 0:
            print(model)
        # build optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(args.epochs * 0.25),
                int(args.epochs * 0.5),
                int(args.epochs * 0.75),
            ],
            gamma=0.3,
        )
        train_metric = metric_cls(True)
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
        model.analyze_weights()
        with torch.no_grad():
            train_metric(preds, historical_y)
        train_results = train_metric.get_info()

        xmetric = ComposeMetric(metric_cls(True), SaveMetric())
        eval_dataset = torch.utils.data.TensorDataset(
            future_x.to(args.device), future_y.to(args.device)
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        results = basic_eval_fn(eval_loader, model, xmetric, logger)
        log_str = (
            "[{:}]".format(time_string())
            + " [{:04d}/{:04d}]".format(idx, len(env))
            + " train-score: {:.5f}, eval-score: {:.5f}".format(
                train_results["score"], results["score"]
            )
        )
        logger.log(log_str)

        save_path = logger.path(None) / "{:04d}-{:04d}.pth".format(idx, len(env))
        w_containers[idx] = model.get_w_container().no_grad_clone()
        save_checkpoint(
            {
                "model_state_dict": model.state_dict(),
                "model": model,
                "index": idx,
                "timestamp": future_time.item(),
            },
            save_path,
            logger,
        )
        logger.log("")
        per_timestamp_time.update(time.time() - start_time)
        start_time = time.time()

    save_checkpoint(
        {"w_containers": w_containers},
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
        default="./outputs/GeMOSA-synthetic/use-same-timestamp",
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
        default=300,
        help="The total number of epochs.",
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
    args.save_dir = "{:}-d{:}_e{:}_lr{:}-env{:}".format(
        args.save_dir, args.hidden_dim, args.epochs, args.init_lr, args.env_version
    )
    main(args)
