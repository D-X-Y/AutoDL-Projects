#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/lfna-ttss-hpnet.py --env_version v1 --hidden_dim 16
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
from lfna_models import HyperNet_VX as HyperNet


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

    criterion = torch.nn.MSELoss()
    logger.log("There are {:} weights.".format(model.get_w_container().numel()))

    # pre-train the model
    dataset = init_dataset = TimeData(0, env_info["0-x"], env_info["0-y"])

    shape_container = model.get_w_container().to_shape_container()
    hypernet = HyperNet(shape_container, 16)
    print(hypernet)

    optimizer = torch.optim.Adam(hypernet.parameters(), lr=args.init_lr, amsgrad=True)

    best_loss, best_param = None, None
    for _iepoch in range(args.epochs):
        container = hypernet(None)

        preds = model.forward_with_container(dataset.x, container)
        optimizer.zero_grad()
        loss = criterion(preds, dataset.y)
        loss.backward()
        optimizer.step()
        # save best
        if best_loss is None or best_loss > loss.item():
            best_loss = loss.item()
            best_param = copy.deepcopy(model.state_dict())
    print("hyper-net : best={:.4f}".format(best_loss))

    init_loss = train_model(model, init_dataset, args.init_lr, args.epochs)
    logger.log("The pre-training loss is {:.4f}".format(init_loss))

    print(model)
    print(hypernet)

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
