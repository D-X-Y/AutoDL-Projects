#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import os, sys, time, torch, random, argparse
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy

from xautodl.config_utils import load_config, dict2config
from xautodl.procedures import get_procedures, get_optim_scheduler
from xautodl.datasets import get_datasets
from xautodl.models import obtain_model
from xautodl.utils import get_model_infos
from xautodl.log_utils import PrintLogger, time_string


def main(args):

    assert os.path.isdir(args.data_path), "invalid data-path : {:}".format(
        args.data_path
    )
    assert os.path.isfile(args.checkpoint), "invalid checkpoint : {:}".format(
        args.checkpoint
    )

    checkpoint = torch.load(args.checkpoint)
    xargs = checkpoint["args"]
    train_data, valid_data, xshape, class_num = get_datasets(
        xargs.dataset, args.data_path, xargs.cutout_length
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=xargs.batch_size,
        shuffle=False,
        num_workers=xargs.workers,
        pin_memory=True,
    )

    logger = PrintLogger()
    model_config = dict2config(checkpoint["model-config"], logger)
    base_model = obtain_model(model_config)
    flop, param = get_model_infos(base_model, xshape)
    logger.log("model ====>>>>:\n{:}".format(base_model))
    logger.log("model information : {:}".format(base_model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )
    logger.log("-" * 50)
    logger.log("valid_data : {:}".format(valid_data))
    optim_config = dict2config(checkpoint["optim-config"], logger)
    _, _, criterion = get_optim_scheduler(base_model.parameters(), optim_config)
    logger.log("criterion  : {:}".format(criterion))
    base_model.load_state_dict(checkpoint["base-model"])
    _, valid_func = get_procedures(xargs.procedure)
    logger.log("initialize the CNN done, evaluate it using {:}".format(valid_func))
    network = torch.nn.DataParallel(base_model).cuda()

    try:
        valid_loss, valid_acc1, valid_acc5 = valid_func(
            valid_loader,
            network,
            criterion,
            optim_config,
            "pure-evaluation",
            xargs.print_freq_eval,
            logger,
        )
    except:
        _, valid_func = get_procedures("basic")
        valid_loss, valid_acc1, valid_acc5 = valid_func(
            valid_loader,
            network,
            criterion,
            optim_config,
            "pure-evaluation",
            xargs.print_freq_eval,
            logger,
        )

    num_bytes = torch.cuda.max_memory_cached(next(network.parameters()).device) * 1.0
    logger.log(
        "***{:s}*** EVALUATION loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),
            valid_loss,
            valid_acc1,
            valid_acc5,
            100 - valid_acc1,
            100 - valid_acc5,
        )
    )
    logger.log(
        "[GPU-Memory-Usage on {:} is {:} bytes, {:.2f} KB, {:.2f} MB, {:.2f} GB.]".format(
            next(network.parameters()).device,
            int(num_bytes),
            num_bytes / 1e3,
            num_bytes / 1e6,
            num_bytes / 1e9,
        )
    )
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate-CNN")
    parser.add_argument("--data_path", type=str, help="Path to dataset.")
    parser.add_argument(
        "--checkpoint", type=str, help="Choose between Cifar10/100 and ImageNet."
    )
    args = parser.parse_args()
    assert torch.cuda.is_available(), "torch.cuda is not available"
    main(args)
