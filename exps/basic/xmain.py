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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    logger.log("The training loader: {:}".format(train_loader))
    logger.log("The validation loader: {:}".format(valid_loader))
    optimizer = xmisc.nested_call_by_yaml(
        args.optim_config,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss = xmisc.nested_call_by_yaml(args.loss_config)

    logger.log("The optimizer is:\n{:}".format(optimizer))
    logger.log("The loss is {:}".format(loss))

    model, loss = torch.nn.DataParallel(model).cuda(), loss.cuda()

    import pdb

    pdb.set_trace()

    train_func, valid_func = get_procedures(args.procedure)

    total_epoch = optim_config.epochs + optim_config.warmup
    # Main Training and Evaluation Loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(start_epoch, total_epoch):
        scheduler.update(epoch, 0.0)
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.avg * (total_epoch - epoch), True)
        )
        epoch_str = "epoch={:03d}/{:03d}".format(epoch, total_epoch)
        LRs = scheduler.get_lr()
        find_best = False
        # set-up drop-out ratio
        if hasattr(base_model, "update_drop_path"):
            base_model.update_drop_path(
                model_config.drop_path_prob * epoch / total_epoch
            )
        logger.log(
            "\n***{:s}*** start {:s} {:s}, LR=[{:.6f} ~ {:.6f}], scheduler={:}".format(
                time_string(), epoch_str, need_time, min(LRs), max(LRs), scheduler
            )
        )

        # train for one epoch
        train_loss, train_acc1, train_acc5 = train_func(
            train_loader,
            network,
            criterion,
            scheduler,
            optimizer,
            optim_config,
            epoch_str,
            args.print_freq,
            logger,
        )
        # log the results
        logger.log(
            "***{:s}*** TRAIN [{:}] loss = {:.6f}, accuracy-1 = {:.2f}, accuracy-5 = {:.2f}".format(
                time_string(), epoch_str, train_loss, train_acc1, train_acc5
            )
        )

        # evaluate the performance
        if (epoch % args.eval_frequency == 0) or (epoch + 1 == total_epoch):
            logger.log("-" * 150)
            valid_loss, valid_acc1, valid_acc5 = valid_func(
                valid_loader,
                network,
                criterion,
                optim_config,
                epoch_str,
                args.print_freq_eval,
                logger,
            )
            valid_accuracies[epoch] = valid_acc1
            logger.log(
                "***{:s}*** VALID [{:}] loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f} | Best-Valid-Acc@1={:.2f}, Error@1={:.2f}".format(
                    time_string(),
                    epoch_str,
                    valid_loss,
                    valid_acc1,
                    valid_acc5,
                    valid_accuracies["best"],
                    100 - valid_accuracies["best"],
                )
            )
            if valid_acc1 > valid_accuracies["best"]:
                valid_accuracies["best"] = valid_acc1
                find_best = True
                logger.log(
                    "Currently, the best validation accuracy found at {:03d}-epoch :: acc@1={:.2f}, acc@5={:.2f}, error@1={:.2f}, error@5={:.2f}, save into {:}.".format(
                        epoch,
                        valid_acc1,
                        valid_acc5,
                        100 - valid_acc1,
                        100 - valid_acc5,
                        model_best_path,
                    )
                )
            num_bytes = (
                torch.cuda.max_memory_cached(next(network.parameters()).device) * 1.0
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
            max_bytes[epoch] = num_bytes
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        # save checkpoint
        save_path = save_checkpoint(
            {
                "epoch": epoch,
                "args": deepcopy(args),
                "max_bytes": deepcopy(max_bytes),
                "FLOP": flop,
                "PARAM": param,
                "valid_accuracies": deepcopy(valid_accuracies),
                "model-config": model_config._asdict(),
                "optim-config": optim_config._asdict(),
                "base-model": base_model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            model_base_path,
            logger,
        )
        if find_best:
            copy_checkpoint(model_base_path, model_best_path, logger)
        last_info = save_checkpoint(
            {
                "epoch": epoch,
                "args": deepcopy(args),
                "last_checkpoint": save_path,
            },
            logger.path("info"),
            logger,
        )

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("\n" + "-" * 200)
    logger.log(
        "Finish training/validation in {:} with Max-GPU-Memory of {:.2f} MB, and save final checkpoint into {:}".format(
            convert_secs2time(epoch_time.sum, True),
            max(v for k, v in max_bytes.items()) / 1e6,
            logger.path("info"),
        )
    )
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
    parser.add_argument(
        "--train_data_config", type=str, help="The training dataset config path."
    )
    parser.add_argument(
        "--valid_data_config", type=str, help="The validation dataset config path."
    )
    parser.add_argument("--data_path", type=str, help="The path to the dataset.")
    parser.add_argument("--algorithm", type=str, help="The algorithm.")
    # Optimization options
    parser.add_argument("--lr", type=float, help="The learning rate")
    parser.add_argument("--weight_decay", type=float, help="The weight decay")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size.")
    parser.add_argument("--workers", type=int, default=4, help="The number of workers")
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")

    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    if args.save_dir is None:
        raise ValueError("The save-path argument can not be None")

    main(args)
