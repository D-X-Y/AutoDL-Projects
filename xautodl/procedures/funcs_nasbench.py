#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
import os, time, copy, torch, pathlib

from xautodl import datasets
from xautodl.config_utils import load_config
from xautodl.procedures import prepare_seed, get_optim_scheduler
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net
from xautodl.utils import get_model_infos
from xautodl.procedures.eval_funcs import obtain_accuracy


__all__ = ["evaluate_for_seed", "pure_evaluate", "get_nas_bench_loaders"]


def pure_evaluate(xloader, network, criterion=torch.nn.CrossEntropyLoss()):
    data_time, batch_time, batch = AverageMeter(), AverageMeter(), None
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    latencies, device = [], torch.cuda.current_device()
    network.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(xloader):
            targets = targets.cuda(device=device, non_blocking=True)
            inputs = inputs.cuda(device=device, non_blocking=True)
            data_time.update(time.time() - end)
            # forward
            features, logits = network(inputs)
            loss = criterion(logits, targets)
            batch_time.update(time.time() - end)
            if batch is None or batch == inputs.size(0):
                batch = inputs.size(0)
                latencies.append(batch_time.val - data_time.val)
            # record loss and accuracy
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            end = time.time()
    if len(latencies) > 2:
        latencies = latencies[1:]
    return losses.avg, top1.avg, top5.avg, latencies


def procedure(xloader, network, criterion, scheduler, optimizer, mode: str):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    if mode == "train":
        network.train()
    elif mode == "valid":
        network.eval()
    else:
        raise ValueError("The mode is not right : {:}".format(mode))
    device = torch.cuda.current_device()
    data_time, batch_time, end = AverageMeter(), AverageMeter(), time.time()
    for i, (inputs, targets) in enumerate(xloader):
        if mode == "train":
            scheduler.update(None, 1.0 * i / len(xloader))

        targets = targets.cuda(device=device, non_blocking=True)
        if mode == "train":
            optimizer.zero_grad()
        # forward
        features, logits = network(inputs)
        loss = criterion(logits, targets)
        # backward
        if mode == "train":
            loss.backward()
            optimizer.step()
        # record loss and accuracy
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # count time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg, top1.avg, top5.avg, batch_time.sum


def evaluate_for_seed(
    arch_config, opt_config, train_loader, valid_loaders, seed: int, logger
):
    """A modular function to train and evaluate a single network, using the given random seed and optimization config with the provided loaders."""
    prepare_seed(seed)  # random seed
    net = get_cell_based_tiny_net(arch_config)
    # net = TinyNetwork(arch_config['channel'], arch_config['num_cells'], arch, config.class_num)
    flop, param = get_model_infos(net, opt_config.xshape)
    logger.log("Network : {:}".format(net.get_message()), False)
    logger.log(
        "{:} Seed-------------------------- {:} --------------------------".format(
            time_string(), seed
        )
    )
    logger.log("FLOP = {:} MB, Param = {:} MB".format(flop, param))
    # train and valid
    optimizer, scheduler, criterion = get_optim_scheduler(net.parameters(), opt_config)
    default_device = torch.cuda.current_device()
    network = torch.nn.DataParallel(net, device_ids=[default_device]).cuda(
        device=default_device
    )
    criterion = criterion.cuda(device=default_device)
    # start training
    start_time, epoch_time, total_epoch = (
        time.time(),
        AverageMeter(),
        opt_config.epochs + opt_config.warmup,
    )
    (
        train_losses,
        train_acc1es,
        train_acc5es,
        valid_losses,
        valid_acc1es,
        valid_acc5es,
    ) = ({}, {}, {}, {}, {}, {})
    train_times, valid_times, lrs = {}, {}, {}
    for epoch in range(total_epoch):
        scheduler.update(epoch, 0.0)
        lr = min(scheduler.get_lr())
        train_loss, train_acc1, train_acc5, train_tm = procedure(
            train_loader, network, criterion, scheduler, optimizer, "train"
        )
        train_losses[epoch] = train_loss
        train_acc1es[epoch] = train_acc1
        train_acc5es[epoch] = train_acc5
        train_times[epoch] = train_tm
        lrs[epoch] = lr
        with torch.no_grad():
            for key, xloder in valid_loaders.items():
                valid_loss, valid_acc1, valid_acc5, valid_tm = procedure(
                    xloder, network, criterion, None, None, "valid"
                )
                valid_losses["{:}@{:}".format(key, epoch)] = valid_loss
                valid_acc1es["{:}@{:}".format(key, epoch)] = valid_acc1
                valid_acc5es["{:}@{:}".format(key, epoch)] = valid_acc5
                valid_times["{:}@{:}".format(key, epoch)] = valid_tm

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.avg * (total_epoch - epoch - 1), True)
        )
        logger.log(
            "{:} {:} epoch={:03d}/{:03d} :: Train [loss={:.5f}, acc@1={:.2f}%, acc@5={:.2f}%] Valid [loss={:.5f}, acc@1={:.2f}%, acc@5={:.2f}%], lr={:}".format(
                time_string(),
                need_time,
                epoch,
                total_epoch,
                train_loss,
                train_acc1,
                train_acc5,
                valid_loss,
                valid_acc1,
                valid_acc5,
                lr,
            )
        )
    info_seed = {
        "flop": flop,
        "param": param,
        "arch_config": arch_config._asdict(),
        "opt_config": opt_config._asdict(),
        "total_epoch": total_epoch,
        "train_losses": train_losses,
        "train_acc1es": train_acc1es,
        "train_acc5es": train_acc5es,
        "train_times": train_times,
        "valid_losses": valid_losses,
        "valid_acc1es": valid_acc1es,
        "valid_acc5es": valid_acc5es,
        "valid_times": valid_times,
        "learning_rates": lrs,
        "net_state_dict": net.state_dict(),
        "net_string": "{:}".format(net),
        "finish-train": True,
    }
    return info_seed


def get_nas_bench_loaders(workers):

    torch.set_num_threads(workers)

    root_dir = (pathlib.Path(__file__).parent / ".." / "..").resolve()
    torch_dir = pathlib.Path(os.environ["TORCH_HOME"])
    # cifar
    cifar_config_path = root_dir / "configs" / "nas-benchmark" / "CIFAR.config"
    cifar_config = load_config(cifar_config_path, None, None)
    get_datasets = datasets.get_datasets  # a function to return the dataset
    break_line = "-" * 150
    print("{:} Create data-loader for all datasets".format(time_string()))
    print(break_line)
    TRAIN_CIFAR10, VALID_CIFAR10, xshape, class_num = get_datasets(
        "cifar10", str(torch_dir / "cifar.python"), -1
    )
    print(
        "original CIFAR-10 : {:} training images and {:} test images : {:} input shape : {:} number of classes".format(
            len(TRAIN_CIFAR10), len(VALID_CIFAR10), xshape, class_num
        )
    )
    cifar10_splits = load_config(
        root_dir / "configs" / "nas-benchmark" / "cifar-split.txt", None, None
    )
    assert cifar10_splits.train[:10] == [
        0,
        5,
        7,
        11,
        13,
        15,
        16,
        17,
        20,
        24,
    ] and cifar10_splits.valid[:10] == [
        1,
        2,
        3,
        4,
        6,
        8,
        9,
        10,
        12,
        14,
    ]
    temp_dataset = copy.deepcopy(TRAIN_CIFAR10)
    temp_dataset.transform = VALID_CIFAR10.transform
    # data loader
    trainval_cifar10_loader = torch.utils.data.DataLoader(
        TRAIN_CIFAR10,
        batch_size=cifar_config.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    train_cifar10_loader = torch.utils.data.DataLoader(
        TRAIN_CIFAR10,
        batch_size=cifar_config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar10_splits.train),
        num_workers=workers,
        pin_memory=True,
    )
    valid_cifar10_loader = torch.utils.data.DataLoader(
        temp_dataset,
        batch_size=cifar_config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar10_splits.valid),
        num_workers=workers,
        pin_memory=True,
    )
    test__cifar10_loader = torch.utils.data.DataLoader(
        VALID_CIFAR10,
        batch_size=cifar_config.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    print(
        "CIFAR-10  : trval-loader has {:3d} batch with {:} per batch".format(
            len(trainval_cifar10_loader), cifar_config.batch_size
        )
    )
    print(
        "CIFAR-10  : train-loader has {:3d} batch with {:} per batch".format(
            len(train_cifar10_loader), cifar_config.batch_size
        )
    )
    print(
        "CIFAR-10  : valid-loader has {:3d} batch with {:} per batch".format(
            len(valid_cifar10_loader), cifar_config.batch_size
        )
    )
    print(
        "CIFAR-10  : test--loader has {:3d} batch with {:} per batch".format(
            len(test__cifar10_loader), cifar_config.batch_size
        )
    )
    print(break_line)
    # CIFAR-100
    TRAIN_CIFAR100, VALID_CIFAR100, xshape, class_num = get_datasets(
        "cifar100", str(torch_dir / "cifar.python"), -1
    )
    print(
        "original CIFAR-100: {:} training images and {:} test images : {:} input shape : {:} number of classes".format(
            len(TRAIN_CIFAR100), len(VALID_CIFAR100), xshape, class_num
        )
    )
    cifar100_splits = load_config(
        root_dir / "configs" / "nas-benchmark" / "cifar100-test-split.txt", None, None
    )
    assert cifar100_splits.xvalid[:10] == [
        1,
        3,
        4,
        5,
        8,
        10,
        13,
        14,
        15,
        16,
    ] and cifar100_splits.xtest[:10] == [
        0,
        2,
        6,
        7,
        9,
        11,
        12,
        17,
        20,
        24,
    ]
    train_cifar100_loader = torch.utils.data.DataLoader(
        TRAIN_CIFAR100,
        batch_size=cifar_config.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    valid_cifar100_loader = torch.utils.data.DataLoader(
        VALID_CIFAR100,
        batch_size=cifar_config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar100_splits.xvalid),
        num_workers=workers,
        pin_memory=True,
    )
    test__cifar100_loader = torch.utils.data.DataLoader(
        VALID_CIFAR100,
        batch_size=cifar_config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar100_splits.xtest),
        num_workers=workers,
        pin_memory=True,
    )
    print(
        "CIFAR-100  : train-loader has {:3d} batch".format(len(train_cifar100_loader))
    )
    print(
        "CIFAR-100  : valid-loader has {:3d} batch".format(len(valid_cifar100_loader))
    )
    print(
        "CIFAR-100  : test--loader has {:3d} batch".format(len(test__cifar100_loader))
    )
    print(break_line)

    imagenet16_config_path = "configs/nas-benchmark/ImageNet-16.config"
    imagenet16_config = load_config(imagenet16_config_path, None, None)
    TRAIN_ImageNet16_120, VALID_ImageNet16_120, xshape, class_num = get_datasets(
        "ImageNet16-120", str(torch_dir / "cifar.python" / "ImageNet16"), -1
    )
    print(
        "original TRAIN_ImageNet16_120: {:} training images and {:} test images : {:} input shape : {:} number of classes".format(
            len(TRAIN_ImageNet16_120), len(VALID_ImageNet16_120), xshape, class_num
        )
    )
    imagenet_splits = load_config(
        root_dir / "configs" / "nas-benchmark" / "imagenet-16-120-test-split.txt",
        None,
        None,
    )
    assert imagenet_splits.xvalid[:10] == [
        1,
        2,
        3,
        6,
        7,
        8,
        9,
        12,
        16,
        18,
    ] and imagenet_splits.xtest[:10] == [
        0,
        4,
        5,
        10,
        11,
        13,
        14,
        15,
        17,
        20,
    ]
    train_imagenet_loader = torch.utils.data.DataLoader(
        TRAIN_ImageNet16_120,
        batch_size=imagenet16_config.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    valid_imagenet_loader = torch.utils.data.DataLoader(
        VALID_ImageNet16_120,
        batch_size=imagenet16_config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet_splits.xvalid),
        num_workers=workers,
        pin_memory=True,
    )
    test__imagenet_loader = torch.utils.data.DataLoader(
        VALID_ImageNet16_120,
        batch_size=imagenet16_config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet_splits.xtest),
        num_workers=workers,
        pin_memory=True,
    )
    print(
        "ImageNet-16-120  : train-loader has {:3d} batch with {:} per batch".format(
            len(train_imagenet_loader), imagenet16_config.batch_size
        )
    )
    print(
        "ImageNet-16-120  : valid-loader has {:3d} batch with {:} per batch".format(
            len(valid_imagenet_loader), imagenet16_config.batch_size
        )
    )
    print(
        "ImageNet-16-120  : test--loader has {:3d} batch with {:} per batch".format(
            len(test__imagenet_loader), imagenet16_config.batch_size
        )
    )

    # 'cifar10', 'cifar100', 'ImageNet16-120'
    loaders = {
        "cifar10@trainval": trainval_cifar10_loader,
        "cifar10@train": train_cifar10_loader,
        "cifar10@valid": valid_cifar10_loader,
        "cifar10@test": test__cifar10_loader,
        "cifar100@train": train_cifar100_loader,
        "cifar100@valid": valid_cifar100_loader,
        "cifar100@test": test__cifar100_loader,
        "ImageNet16-120@train": train_imagenet_loader,
        "ImageNet16-120@valid": valid_imagenet_loader,
        "ImageNet16-120@test": test__imagenet_loader,
    }
    return loaders
