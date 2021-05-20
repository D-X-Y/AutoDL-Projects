#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#######################################################################
# Network Pruning via Transformable Architecture Search, NeurIPS 2019 #
#######################################################################
import sys, time, torch, random, argparse
from PIL import ImageFile
from os import path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from copy import deepcopy
from pathlib import Path

from xautodl.config_utils import (
    load_config,
    configure2str,
    obtain_search_args as obtain_args,
)
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
)
from xautodl.procedures import get_optim_scheduler, get_procedures
from xautodl.datasets import get_datasets, SearchDataset
from xautodl.models import obtain_search_model, obtain_model, change_key
from xautodl.utils import get_model_infos
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)

    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)

    # prepare dataset
    train_data, valid_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True , num_workers=args.workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    split_file_path = Path(args.split_path)
    assert split_file_path.exists(), "{:} does not exist".format(split_file_path)
    split_info = torch.load(split_file_path)

    train_split, valid_split = split_info["train"], split_info["valid"]
    assert (
        len(set(train_split).intersection(set(valid_split))) == 0
    ), "There should be 0 element that belongs to both train and valid"
    assert len(train_split) + len(valid_split) == len(
        train_data
    ), "{:} + {:} vs {:}".format(len(train_split), len(valid_split), len(train_data))
    search_dataset = SearchDataset(args.dataset, train_data, train_split, valid_split)

    search_train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
        pin_memory=True,
        num_workers=args.workers,
    )
    search_valid_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
        pin_memory=True,
        num_workers=args.workers,
    )
    search_loader = torch.utils.data.DataLoader(
        search_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )
    # get configures
    if args.ablation_num_select is None or args.ablation_num_select <= 0:
        model_config = load_config(
            args.model_config, {"class_num": class_num, "search_mode": "shape"}, logger
        )
    else:
        model_config = load_config(
            args.model_config,
            {
                "class_num": class_num,
                "search_mode": "ablation",
                "num_random_select": args.ablation_num_select,
            },
            logger,
        )

    # obtain the model
    search_model = obtain_search_model(model_config)
    MAX_FLOP, param = get_model_infos(search_model, xshape)
    optim_config = load_config(
        args.optim_config, {"class_num": class_num, "FLOP": MAX_FLOP}, logger
    )
    logger.log("Model Information : {:}".format(search_model.get_message()))
    logger.log("MAX_FLOP = {:} M".format(MAX_FLOP))
    logger.log("Params   = {:} M".format(param))
    logger.log("train_data : {:}".format(train_data))
    logger.log("search-data: {:}".format(search_dataset))
    logger.log("search_train_loader : {:} samples".format(len(train_split)))
    logger.log("search_valid_loader : {:} samples".format(len(valid_split)))
    base_optimizer, scheduler, criterion = get_optim_scheduler(
        search_model.base_parameters(), optim_config
    )
    arch_optimizer = torch.optim.Adam(
        search_model.arch_parameters(optim_config.arch_LR),
        lr=optim_config.arch_LR,
        betas=(0.5, 0.999),
        weight_decay=optim_config.arch_decay,
    )
    logger.log("base-optimizer : {:}".format(base_optimizer))
    logger.log("arch-optimizer : {:}".format(arch_optimizer))
    logger.log("scheduler      : {:}".format(scheduler))
    logger.log("criterion      : {:}".format(criterion))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )
    network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

    # load checkpoint
    if last_info.exists() or (
        args.resume is not None and osp.isfile(args.resume)
    ):  # automatically resume from previous checkpoint
        if args.resume is not None and osp.isfile(args.resume):
            resume_path = Path(args.resume)
        elif last_info.exists():
            resume_path = last_info
        else:
            raise ValueError("Something is wrong.")
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(resume_path)
        )
        checkpoint = torch.load(resume_path)
        if "last_checkpoint" in checkpoint:
            last_checkpoint_path = checkpoint["last_checkpoint"]
            if not last_checkpoint_path.exists():
                logger.log(
                    "Does not find {:}, try another path".format(last_checkpoint_path)
                )
                last_checkpoint_path = (
                    resume_path.parent
                    / last_checkpoint_path.parent.name
                    / last_checkpoint_path.name
                )
            assert (
                last_checkpoint_path.exists()
            ), "can not find the checkpoint from {:}".format(last_checkpoint_path)
            checkpoint = torch.load(last_checkpoint_path)
        start_epoch = checkpoint["epoch"] + 1
        # for key, value in checkpoint['search_model'].items():
        #  print('K {:} = Shape={:}'.format(key, value.shape))
        search_model.load_state_dict(checkpoint["search_model"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        base_optimizer.load_state_dict(checkpoint["base_optimizer"])
        arch_optimizer.load_state_dict(checkpoint["arch_optimizer"])
        valid_accuracies = checkpoint["valid_accuracies"]
        arch_genotypes = checkpoint["arch_genotypes"]
        discrepancies = checkpoint["discrepancies"]
        max_bytes = checkpoint["max_bytes"]
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                resume_path, start_epoch
            )
        )
    else:
        logger.log(
            "=> do not find the last-info file : {:} or resume : {:}".format(
                last_info, args.resume
            )
        )
        start_epoch, valid_accuracies, arch_genotypes, discrepancies, max_bytes = (
            0,
            {"best": -1},
            {},
            {},
            {},
        )

    # main procedure
    train_func, valid_func = get_procedures(args.procedure)
    total_epoch = optim_config.epochs + optim_config.warmup
    start_time, epoch_time = time.time(), AverageMeter()
    for epoch in range(start_epoch, total_epoch):
        scheduler.update(epoch, 0.0)
        search_model.set_tau(
            args.gumbel_tau_max, args.gumbel_tau_min, epoch * 1.0 / total_epoch
        )
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.avg * (total_epoch - epoch), True)
        )
        epoch_str = "epoch={:03d}/{:03d}".format(epoch, total_epoch)
        LRs = scheduler.get_lr()
        find_best = False

        logger.log(
            "\n***{:s}*** start {:s} {:s}, LR=[{:.6f} ~ {:.6f}], scheduler={:}, tau={:}, FLOP={:.2f}".format(
                time_string(),
                epoch_str,
                need_time,
                min(LRs),
                max(LRs),
                scheduler,
                search_model.tau,
                MAX_FLOP,
            )
        )

        # train for one epoch
        train_base_loss, train_arch_loss, train_acc1, train_acc5 = train_func(
            search_loader,
            network,
            criterion,
            scheduler,
            base_optimizer,
            arch_optimizer,
            optim_config,
            {
                "epoch-str": epoch_str,
                "FLOP-exp": MAX_FLOP * args.FLOP_ratio,
                "FLOP-weight": args.FLOP_weight,
                "FLOP-tolerant": MAX_FLOP * args.FLOP_tolerant,
            },
            args.print_freq,
            logger,
        )
        # log the results
        logger.log(
            "***{:s}*** TRAIN [{:}] base-loss = {:.6f}, arch-loss = {:.6f}, accuracy-1 = {:.2f}, accuracy-5 = {:.2f}".format(
                time_string(),
                epoch_str,
                train_base_loss,
                train_arch_loss,
                train_acc1,
                train_acc5,
            )
        )
        cur_FLOP, genotype = search_model.get_flop(
            "genotype", model_config._asdict(), None
        )
        arch_genotypes[epoch] = genotype
        arch_genotypes["last"] = genotype
        logger.log("[{:}] genotype : {:}".format(epoch_str, genotype))
        # save the configuration
        configure2str(
            genotype,
            str(logger.path("log") / "seed-{:}-temp.config".format(args.rand_seed)),
        )
        arch_info, discrepancy = search_model.get_arch_info()
        logger.log(arch_info)
        discrepancies[epoch] = discrepancy
        logger.log(
            "[{:}] FLOP : {:.2f} MB, ratio : {:.4f}, Expected-ratio : {:.4f}, Discrepancy : {:.3f}".format(
                epoch_str,
                cur_FLOP,
                cur_FLOP / MAX_FLOP,
                args.FLOP_ratio,
                np.mean(discrepancy),
            )
        )

        # if cur_FLOP/MAX_FLOP > args.FLOP_ratio:
        #  init_flop_weight = init_flop_weight * args.FLOP_decay
        # else:
        #  init_flop_weight = init_flop_weight / args.FLOP_decay

        # evaluate the performance
        if (epoch % args.eval_frequency == 0) or (epoch + 1 == total_epoch):
            logger.log("-" * 150)
            valid_loss, valid_acc1, valid_acc5 = valid_func(
                search_valid_loader,
                network,
                criterion,
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
                arch_genotypes["best"] = genotype
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
            # log the GPU memory usage
            # num_bytes = torch.cuda.max_memory_allocated( next(network.parameters()).device ) * 1.0
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

        # save checkpoint
        save_path = save_checkpoint(
            {
                "epoch": epoch,
                "args": deepcopy(args),
                "max_bytes": deepcopy(max_bytes),
                "valid_accuracies": deepcopy(valid_accuracies),
                "model-config": model_config._asdict(),
                "optim-config": optim_config._asdict(),
                "search_model": search_model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "base_optimizer": base_optimizer.state_dict(),
                "arch_optimizer": arch_optimizer.state_dict(),
                "arch_genotypes": arch_genotypes,
                "discrepancies": discrepancies,
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

    logger.log("")
    logger.log("-" * 100)
    last_config_path = logger.path("log") / "seed-{:}-last.config".format(
        args.rand_seed
    )
    configure2str(arch_genotypes["last"], str(last_config_path))
    logger.log(
        "save the last config int {:} :\n{:}".format(
            last_config_path, arch_genotypes["last"]
        )
    )

    best_arch, valid_acc = arch_genotypes["best"], valid_accuracies["best"]
    for key, config in arch_genotypes.items():
        if key == "last":
            continue
        FLOP_ratio = config["estimated_FLOP"] / MAX_FLOP
        if abs(FLOP_ratio - args.FLOP_ratio) <= args.FLOP_tolerant:
            if valid_acc <= valid_accuracies[key]:
                best_arch, valid_acc = config, valid_accuracies[key]
    print(
        "Best-Arch : {:}\nRatio={:}, Valid-ACC={:}".format(
            best_arch, best_arch["estimated_FLOP"] / MAX_FLOP, valid_acc
        )
    )
    best_config_path = logger.path("log") / "seed-{:}-best.config".format(
        args.rand_seed
    )
    configure2str(best_arch, str(best_config_path))
    logger.log(
        "save the last config int {:} :\n{:}".format(best_config_path, best_arch)
    )
    logger.log("\n" + "-" * 200)
    logger.log(
        "Finish training/validation in {:} with Max-GPU-Memory of {:.2f} GB, and save final checkpoint into {:}".format(
            convert_secs2time(epoch_time.sum, True),
            max(v for k, v in max_bytes.items()) / 1e9,
            logger.path("info"),
        )
    )
    logger.close()


if __name__ == "__main__":
    args = obtain_args()
    main(args)
