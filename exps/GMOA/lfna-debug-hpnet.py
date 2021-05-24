#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/lfna-debug-hpnet.py --env_version v1 --hidden_dim 16 --meta_batch 64 --device cuda
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
from xlayers import super_core, trunc_normal_


from lfna_utils import lfna_setup, train_model, TimeData

from lfna_models import HyperNet


def main(args):
    logger, env_info, model_kwargs = lfna_setup(args)
    dynamic_env = env_info["dynamic_env"]
    model = get_model(**model_kwargs)
    criterion = torch.nn.MSELoss()

    shape_container = model.get_w_container().to_shape_container()
    hypernet = HyperNet(
        shape_container, args.hidden_dim, args.task_dim, len(dynamic_env)
    )
    hypernet = hypernet.to(args.device)

    logger.log(
        "{:} There are {:} weights in the base-model.".format(
            time_string(), model.numel()
        )
    )
    logger.log(
        "{:} There are {:} weights in the meta-model.".format(
            time_string(), hypernet.numel()
        )
    )

    for i in range(len(dynamic_env)):
        env_info["{:}-x".format(i)] = env_info["{:}-x".format(i)].to(args.device)
        env_info["{:}-y".format(i)] = env_info["{:}-y".format(i)].to(args.device)
    logger.log("{:} Convert to device-{:} done".format(time_string(), args.device))

    optimizer = torch.optim.Adam(
        hypernet.parameters(), lr=args.init_lr, weight_decay=1e-5, amsgrad=True
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(args.epochs * 0.8),
            int(args.epochs * 0.9),
        ],
        gamma=0.1,
    )

    # LFNA meta-training
    per_epoch_time, start_time = AverageMeter(), time.time()
    last_success_epoch = 0
    for iepoch in range(args.epochs):

        need_time = "Time Left: {:}".format(
            convert_secs2time(per_epoch_time.avg * (args.epochs - iepoch), True)
        )
        head_str = (
            "[{:}] [{:04d}/{:04d}] ".format(time_string(), iepoch, args.epochs)
            + need_time
        )
        # One Epoch
        loss_meter = AverageMeter()
        for istep in range(args.per_epoch_step):
            losses = []
            for ibatch in range(args.meta_batch):
                cur_time = random.randint(0, len(dynamic_env) - 1)
                cur_container = hypernet(cur_time)
                cur_x = env_info["{:}-x".format(cur_time)]
                cur_y = env_info["{:}-y".format(cur_time)]
                cur_dataset = TimeData(cur_time, cur_x, cur_y)

                preds = model.forward_with_container(cur_dataset.x, cur_container)
                optimizer.zero_grad()
                loss = criterion(preds, cur_dataset.y)

                losses.append(loss)
            final_loss = torch.stack(losses).mean()
            final_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_meter.update(final_loss.item())
        success, best_score = hypernet.save_best(-loss_meter.avg)
        if success:
            logger.log("Achieve the best with best_score = {:.3f}".format(best_score))
            last_success_epoch = iepoch
        if iepoch - last_success_epoch >= args.early_stop_thresh:
            logger.log("Early stop at {:}".format(iepoch))
            break
        logger.log(
            head_str
            + " meta-loss: {:.4f} ({:.4f}) :: lr={:.5f}, batch={:}".format(
                loss_meter.avg,
                loss_meter.val,
                min(lr_scheduler.get_last_lr()),
                len(losses),
            )
        )

        save_checkpoint(
            {
                "hypernet": hypernet.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "iepoch": iepoch,
            },
            logger.path("model"),
            logger,
        )
        per_epoch_time.update(time.time() - start_time)
        start_time = time.time()

    print(model)
    print(hypernet)
    hypernet.load_best()

    w_container_per_epoch = dict()
    for idx in range(0, env_info["total"]):
        future_x = env_info["{:}-x".format(idx)]
        future_y = env_info["{:}-y".format(idx)]
        future_container = hypernet(idx)
        w_container_per_epoch[idx] = future_container.no_grad_clone()
        with torch.no_grad():
            future_y_hat = model.forward_with_container(
                future_x, w_container_per_epoch[idx]
            )
            future_loss = criterion(future_y_hat, future_y)
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
        default="./outputs/lfna-synthetic/lfna-debug-hpnet",
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
        default=0.01,
        help="The initial learning rate for the optimizer (default is Adam)",
    )
    parser.add_argument(
        "--meta_batch",
        type=int,
        default=64,
        help="The batch size for the meta-model",
    )
    parser.add_argument(
        "--early_stop_thresh",
        type=int,
        default=100,
        help="The maximum epochs for early stop.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="The total number of epochs.",
    )
    parser.add_argument(
        "--per_epoch_step",
        type=int,
        default=20,
        help="The total number of epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "The save dir argument can not be None"
    args.task_dim = args.hidden_dim
    args.save_dir = "{:}-{:}-d{:}".format(
        args.save_dir, args.env_version, args.hidden_dim
    )
    main(args)
