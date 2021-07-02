#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.06 #
#####################################################
# python exps/basic/xmain.py --save_dir outputs/x   #
#####################################################
import os, sys, time, torch, random, argparse
from copy import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "..").resolve()
print("LIB-DIR: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from xautodl import xmisc


def main(args):

    train_data = xmisc.nested_call_by_yaml(args.train_data_config, args.data_path)
    valid_data = xmisc.nested_call_by_yaml(args.valid_data_config, args.data_path)
    logger = xmisc.Logger(args.save_dir, prefix="seed-{:}-".format(args.rand_seed))

    logger.log("Create the logger: {:}".format(logger))
    logger.log("Arguments : -------------------------------")
    for name, value in args._get_kwargs():
        logger.log("{:16} : {:}".format(name, value))
    logger.log("Python  Version  : {:}".format(sys.version.replace("\n", " ")))
    logger.log("PyTorch Version  : {:}".format(torch.__version__))
    logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logger.log(
        "CUDA_VISIBLE_DEVICES : {:}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else "None"
        )
    )
    logger.log("The training data is:\n{:}".format(train_data))
    logger.log("The validation data is:\n{:}".format(valid_data))

    model = xmisc.nested_call_by_yaml(args.model_config)
    logger.log("The model is:\n{:}".format(model))
    logger.log("The model size is {:.4f} M".format(xmisc.count_parameters(model)))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_sampler=xmisc.BatchSampler(train_data, args.batch_size, args.steps),
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    iters_per_epoch = len(train_data) // args.batch_size

    logger.log("The training loader: {:}".format(train_loader))
    logger.log("The validation loader: {:}".format(valid_loader))
    optimizer = xmisc.nested_call_by_yaml(
        args.optim_config,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    objective = xmisc.nested_call_by_yaml(args.loss_config)
    metric = xmisc.nested_call_by_yaml(args.metric_config)

    logger.log("The optimizer is:\n{:}".format(optimizer))
    logger.log("The objective is {:}".format(objective))
    logger.log("The metric is {:}".format(metric))
    logger.log(
        "The iters_per_epoch = {:}, estimated epochs = {:}".format(
            iters_per_epoch, args.steps // iters_per_epoch
        )
    )

    model, objective = torch.nn.DataParallel(model).cuda(), objective.cuda()
    scheduler = xmisc.LRMultiplier(
        optimizer, xmisc.get_scheduler(args.scheduler, args.lr), args.steps
    )

    start_time, iter_time = time.time(), xmisc.AverageMeter()
    for xiter, data in enumerate(train_loader):
        need_time = "Time Left: {:}".format(
            xmisc.time_utils.convert_secs2time(
                iter_time.avg * (len(train_loader) - xiter), True
            )
        )
        iter_str = "{:6d}/{:06d}".format(xiter, len(train_loader))

        inputs, targets = data
        targets = targets.cuda(non_blocking=True)
        model.train()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = objective(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if xiter % iters_per_epoch == 0:
            logger.log("TRAIN [{:}] loss = {:.6f}".format(iter_str, loss.item()))

        # measure elapsed time
        iter_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("-" * 200 + "\n")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classification model with a loss function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_dir", type=str, help="Folder to save checkpoints and log."
    )
    parser.add_argument("--resume", type=str, help="Resume path.")
    parser.add_argument("--init_model", type=str, help="The initialization model path.")
    parser.add_argument("--model_config", type=str, help="The path to the model config")
    parser.add_argument("--optim_config", type=str, help="The optimizer config file.")
    parser.add_argument("--loss_config", type=str, help="The loss config file.")
    parser.add_argument("--metric_config", type=str, help="The metric config file.")
    parser.add_argument(
        "--train_data_config", type=str, help="The training dataset config path."
    )
    parser.add_argument(
        "--valid_data_config", type=str, help="The validation dataset config path."
    )
    parser.add_argument("--data_path", type=str, help="The path to the dataset.")
    # Optimization options
    parser.add_argument("--lr", type=float, help="The learning rate")
    parser.add_argument("--weight_decay", type=float, help="The weight decay")
    parser.add_argument("--scheduler", type=str, help="The scheduler indicator.")
    parser.add_argument("--steps", type=int, help="The total number of steps.")
    parser.add_argument("--batch_size", type=int, default=256, help="The batch size.")
    parser.add_argument("--workers", type=int, default=4, help="The number of workers")
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")

    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    if args.save_dir is None:
        raise ValueError("The save-path argument can not be None")

    main(args)
