#####################################################
# Learning to Generate Model One Step Ahead         #
#####################################################
# python exps/GeMOSA/lfna.py --env_version v1 --workers 0
# python exps/GeMOSA/lfna.py --env_version v1 --device cuda --lr 0.001
# python exps/GeMOSA/main.py --env_version v1 --device cuda --lr 0.002 --seq_length 16 --meta_batch 128
# python exps/GeMOSA/lfna.py --env_version v1 --device cuda --lr 0.002 --seq_length 24 --time_dim 32 --meta_batch 128
#####################################################
import sys, time, copy, torch, random, argparse
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from torch.nn import functional as F

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

from xautodl.procedures.advanced_main import basic_train_fn, basic_eval_fn
from xautodl.procedures.metric_utils import SaveMetric, MSEMetric, ComposeMetric
from xautodl.datasets.synthetic_core import get_synthetic_env
from xautodl.models.xcore import get_model
from xautodl.xlayers import super_core, trunc_normal_

from lfna_utils import lfna_setup, train_model, TimeData
from lfna_meta_model import MetaModelV1


def online_evaluate(env, meta_model, base_model, criterion, args, logger, save=False):
    logger.log("Online evaluate: {:}".format(env))
    loss_meter = AverageMeter()
    w_containers = dict()
    for idx, (future_time, (future_x, future_y)) in enumerate(env):
        with torch.no_grad():
            meta_model.eval()
            base_model.eval()
            [future_container], time_embeds = meta_model(
                future_time.to(args.device).view(-1), None, False
            )
            if save:
                w_containers[idx] = future_container.no_grad_clone()
            future_x, future_y = future_x.to(args.device), future_y.to(args.device)
            future_y_hat = base_model.forward_with_container(future_x, future_container)
            future_loss = criterion(future_y_hat, future_y)
            loss_meter.update(future_loss.item())
        refine, post_refine_loss = meta_model.adapt(
            base_model,
            criterion,
            future_time.item(),
            future_x,
            future_y,
            args.refine_lr,
            args.refine_epochs,
            {"param": time_embeds, "loss": future_loss.item()},
        )
        logger.log(
            "[ONLINE] [{:03d}/{:03d}] loss={:.4f}".format(
                idx, len(env), future_loss.item()
            )
            + ", post-loss={:.4f}".format(post_refine_loss if refine else -1)
        )
    meta_model.clear_fixed()
    meta_model.clear_learnt()
    return w_containers, loss_meter


def meta_train_procedure(base_model, meta_model, criterion, xenv, args, logger):
    base_model.train()
    meta_model.train()
    optimizer = torch.optim.Adam(
        meta_model.get_parameters(True, True, True),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    logger.log("Pre-train the meta-model")
    logger.log("Using the optimizer: {:}".format(optimizer))

    meta_model.set_best_dir(logger.path(None) / "ckps-pretrain-v2")
    final_best_name = "final-pretrain-{:}.pth".format(args.rand_seed)
    if meta_model.has_best(final_best_name):
        meta_model.load_best(final_best_name)
        logger.log("Directly load the best model from {:}".format(final_best_name))
        return

    total_indexes = list(range(meta_model.meta_length))
    meta_model.set_best_name("pretrain-{:}.pth".format(args.rand_seed))
    last_success_epoch, early_stop_thresh = 0, args.pretrain_early_stop_thresh
    per_epoch_time, start_time = AverageMeter(), time.time()
    device = args.device
    for iepoch in range(args.epochs):
        left_time = "Time Left: {:}".format(
            convert_secs2time(per_epoch_time.avg * (args.epochs - iepoch), True)
        )
        optimizer.zero_grad()

        generated_time_embeds = meta_model(meta_model.meta_timestamps, None, True)

        batch_indexes = random.choices(total_indexes, k=args.meta_batch)

        raw_time_steps = meta_model.meta_timestamps[batch_indexes]

        regularization_loss = F.l1_loss(
            generated_time_embeds, meta_model.super_meta_embed, reduction="mean"
        )
        # future loss
        total_future_losses, total_present_losses = [], []
        future_containers, _ = meta_model(
            None, generated_time_embeds[batch_indexes], False
        )
        present_containers, _ = meta_model(
            None, meta_model.super_meta_embed[batch_indexes], False
        )
        for ibatch, time_step in enumerate(raw_time_steps.cpu().tolist()):
            _, (inputs, targets) = xenv(time_step)
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = base_model.forward_with_container(
                inputs, future_containers[ibatch]
            )
            total_future_losses.append(criterion(predictions, targets))

            predictions = base_model.forward_with_container(
                inputs, present_containers[ibatch]
            )
            total_present_losses.append(criterion(predictions, targets))

        with torch.no_grad():
            meta_std = torch.stack(total_future_losses).std().item()
        loss_future = torch.stack(total_future_losses).mean()
        loss_present = torch.stack(total_present_losses).mean()
        total_loss = loss_future + loss_present + regularization_loss
        total_loss.backward()
        optimizer.step()
        # success
        success, best_score = meta_model.save_best(-total_loss.item())
        logger.log(
            "{:} [META {:04d}/{:}] loss : {:.4f} +- {:.4f} = {:.4f} + {:.4f} + {:.4f}".format(
                time_string(),
                iepoch,
                args.epochs,
                total_loss.item(),
                meta_std,
                loss_future.item(),
                loss_present.item(),
                regularization_loss.item(),
            )
            + ", batch={:}".format(len(total_future_losses))
            + ", success={:}, best={:.4f}".format(success, -best_score)
            + ", LS={:}/{:}".format(iepoch - last_success_epoch, early_stop_thresh)
            + ", {:}".format(left_time)
        )
        if success:
            last_success_epoch = iepoch
        if iepoch - last_success_epoch >= early_stop_thresh:
            logger.log("Early stop the pre-training at {:}".format(iepoch))
            break
        per_epoch_time.update(time.time() - start_time)
        start_time = time.time()
    meta_model.load_best()
    # save to the final model
    meta_model.set_best_name(final_best_name)
    success, _ = meta_model.save_best(best_score + 1e-6)
    assert success
    logger.log("Save the best model into {:}".format(final_best_name))


def main(args):
    logger, model_kwargs = lfna_setup(args)
    train_env = get_synthetic_env(mode="train", version=args.env_version)
    valid_env = get_synthetic_env(mode="valid", version=args.env_version)
    trainval_env = get_synthetic_env(mode="trainval", version=args.env_version)
    all_env = get_synthetic_env(mode=None, version=args.env_version)
    logger.log("The training enviornment: {:}".format(train_env))
    logger.log("The validation enviornment: {:}".format(valid_env))
    logger.log("The trainval enviornment: {:}".format(trainval_env))
    logger.log("The total enviornment: {:}".format(all_env))

    base_model = get_model(**model_kwargs)
    base_model = base_model.to(args.device)
    criterion = torch.nn.MSELoss()

    shape_container = base_model.get_w_container().to_shape_container()

    # pre-train the hypernetwork
    timestamps = trainval_env.get_timestamp(None)
    meta_model = MetaModelV1(
        shape_container,
        args.layer_dim,
        args.time_dim,
        timestamps,
        seq_length=args.seq_length,
        interval=trainval_env.time_interval,
    )
    meta_model = meta_model.to(args.device)

    logger.log("The base-model has {:} weights.".format(base_model.numel()))
    logger.log("The meta-model has {:} weights.".format(meta_model.numel()))
    logger.log("The base-model is\n{:}".format(base_model))
    logger.log("The meta-model is\n{:}".format(meta_model))

    meta_train_procedure(base_model, meta_model, criterion, trainval_env, args, logger)

    # try to evaluate once
    # online_evaluate(train_env, meta_model, base_model, criterion, args, logger)
    # online_evaluate(valid_env, meta_model, base_model, criterion, args, logger)
    w_containers, loss_meter = online_evaluate(
        all_env, meta_model, base_model, criterion, args, logger, True
    )
    logger.log("In this enviornment, the loss-meter is {:}".format(loss_meter))

    save_checkpoint(
        {"w_containers": w_containers},
        logger.path(None) / "final-ckp.pth",
        logger,
    )

    logger.log("-" * 200 + "\n")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(".")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/lfna-synthetic/lfna-battle",
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
        "--layer_dim",
        type=int,
        default=16,
        help="The layer chunk dimension.",
    )
    parser.add_argument(
        "--time_dim",
        type=int,
        default=16,
        help="The timestamp dimension.",
    )
    #####
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="The initial learning rate for the optimizer (default is Adam)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.00001,
        help="The weight decay for the optimizer (default is Adam)",
    )
    parser.add_argument(
        "--meta_batch",
        type=int,
        default=64,
        help="The batch size for the meta-model",
    )
    parser.add_argument(
        "--sampler_enlarge",
        type=int,
        default=5,
        help="Enlarge the #iterations for an epoch",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="The total #epochs.")
    parser.add_argument(
        "--refine_lr",
        type=float,
        default=0.001,
        help="The learning rate for the optimizer, during refine",
    )
    parser.add_argument(
        "--refine_epochs", type=int, default=150, help="The final refine #epochs."
    )
    parser.add_argument(
        "--early_stop_thresh",
        type=int,
        default=20,
        help="The #epochs for early stop.",
    )
    parser.add_argument(
        "--pretrain_early_stop_thresh",
        type=int,
        default=300,
        help="The #epochs for early stop.",
    )
    parser.add_argument(
        "--seq_length", type=int, default=10, help="The sequence length."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="The number of workers in parallel."
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
    args.save_dir = "{:}-bs{:}-d{:}_{:}_{:}-s{:}-lr{:}-wd{:}-e{:}-env{:}".format(
        args.save_dir,
        args.meta_batch,
        args.hidden_dim,
        args.layer_dim,
        args.time_dim,
        args.seq_length,
        args.lr,
        args.weight_decay,
        args.epochs,
        args.env_version,
    )
    main(args)
