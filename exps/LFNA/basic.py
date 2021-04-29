#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/basic.py
#####################################################
import sys, time, torch, random, argparse
from copy import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from procedures import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint
from log_utils import time_string

from procedures.advanced_main import basic_train_fn, basic_eval_fn
from procedures.metric_utils import SaveMetric, MSEMetric, ComposeMetric
from datasets.synthetic_core import get_synthetic_env
from models.xcore import get_model


def main(args):
    torch.set_num_threads(args.workers)
    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)

    dynamic_env = get_synthetic_env()
    historical_x, historical_y = None, None
    for idx, (timestamp, (allx, ally)) in enumerate(dynamic_env):

        if historical_x is not None:
            mean, std = historical_x.mean().item(), historical_x.std().item()
        else:
            mean, std = 0, 1
        model_kwargs = dict(input_dim=1, output_dim=1, mean=mean, std=std)
        model = get_model(dict(model_type="simple_mlp"), **model_kwargs)

        # create the current data loader
        if historical_x is not None:
            train_dataset = torch.utils.data.TensorDataset(historical_x, historical_y)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.init_lr, amsgrad=True
            )
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
            for _iepoch in range(args.epochs):
                results = basic_train_fn(
                    train_loader, model, criterion, optimizer, MSEMetric(), logger
                )
                lr_scheduler.step()
                if _iepoch % args.log_per_epoch == 0:
                    log_str = (
                        "[{:}]".format(time_string())
                        + " [{:04d}/{:04d}][{:04d}/{:04d}]".format(
                            idx, len(dynamic_env), _iepoch, args.epochs
                        )
                        + " mse: {:.5f}, lr: {:.4f}".format(
                            results["mse"], min(lr_scheduler.get_last_lr())
                        )
                    )
                    logger.log(log_str)
            results = basic_eval_fn(train_loader, model, MSEMetric(), logger)
            logger.log(
                "[{:}] [{:04d}/{:04d}] train-mse: {:.5f}".format(
                    time_string(), idx, len(dynamic_env), results["mse"]
                )
            )

        metric = ComposeMetric(MSEMetric(), SaveMetric())
        eval_dataset = torch.utils.data.TensorDataset(allx, ally)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        results = basic_eval_fn(eval_loader, model, metric, logger)
        log_str = (
            "[{:}]".format(time_string())
            + " [{:04d}/{:04d}]".format(idx, len(dynamic_env))
            + " eval-mse: {:.5f}".format(results["mse"])
        )
        logger.log(log_str)

        save_path = logger.path(None) / "{:04d}-{:04d}.pth".format(
            idx, len(dynamic_env)
        )
        save_checkpoint(
            {"model": model.state_dict(), "index": idx, "timestamp": timestamp},
            save_path,
            logger,
        )

        # Update historical data
        if historical_x is None:
            historical_x, historical_y = allx, ally
        else:
            historical_x, historical_y = torch.cat((historical_x, allx)), torch.cat(
                (historical_y, ally)
            )
        logger.log("")

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
        "--init_lr",
        type=float,
        default=0.1,
        help="The initial learning rate for the optimizer (default is Adam)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="The total number of epochs.",
    )
    parser.add_argument(
        "--log_per_epoch",
        type=int,
        default=200,
        help="Log the training information per __ epochs.",
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
