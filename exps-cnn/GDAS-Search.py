import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from utils import AverageMeter, time_string, convert_secs2time
from utils import print_log, obtain_accuracy
from utils import Cutout, count_parameters_in_MB
from nas import Network, NetworkACC2, NetworkV3, NetworkV4, NetworkV5, NetworkFACC1
from nas import return_alphas_str
from train_utils import main_procedure
from scheduler import load_config

Networks = {'base': Network, 'acc2': NetworkACC2, 'facc1': NetworkFACC1, 'NetworkV3': NetworkV3, 'NetworkV4': NetworkV4, 'NetworkV5': NetworkV5}


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data_path',         type=str,   help='Path to dataset')
parser.add_argument('--dataset',           type=str,   choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch',              type=str,   choices=Networks.keys(),         help='Choose networks.')
parser.add_argument('--batch_size',        type=int,   help='the batch size')
parser.add_argument('--learning_rate_max', type=float, help='initial learning rate')
parser.add_argument('--learning_rate_min', type=float, help='minimum learning rate')
parser.add_argument('--tau_max',           type=float, help='initial tau')
parser.add_argument('--tau_min',           type=float, help='minimum tau')
parser.add_argument('--momentum',          type=float, help='momentum')
parser.add_argument('--weight_decay',      type=float, help='weight decay')
parser.add_argument('--epochs',            type=int,   help='num of training epochs')
# architecture leraning rate
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
#
parser.add_argument('--init_channels',      type=int, help='num of init channels')
parser.add_argument('--layers',             type=int, help='total number of layers')
# 
parser.add_argument('--cutout',         type=int,   help='cutout length, negative means no cutout')
parser.add_argument('--grad_clip',      type=float, help='gradient clipping')
parser.add_argument('--model_config',   type=str  , help='the model configuration')

# resume
parser.add_argument('--resume',         type=str  , help='the resume path')
parser.add_argument('--only_base',action='store_true', default=False, help='only train the searched model')
# split data
parser.add_argument('--validate', action='store_true', default=False, help='split train-data int train/val or not')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
# log
parser.add_argument('--workers',       type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--save_path',     type=str, help='Folder to save checkpoints and log.')
parser.add_argument('--print_freq',    type=int, help='print frequency (default: 200)')
parser.add_argument('--manualSeed',    type=int, help='manual seed')
args = parser.parse_args()

assert torch.cuda.is_available(), 'torch.cuda is not available'

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
cudnn.benchmark = True
cudnn.enabled   = True
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)


def main():

  # Init logger
  args.save_path = os.path.join(args.save_path, 'seed-{:}'.format(args.manualSeed))
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log-seed-{:}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("Torch  version : {}".format(torch.__version__), log)
  print_log("CUDA   version : {}".format(torch.version.cuda), log)
  print_log("cuDNN  version : {}".format(cudnn.version()), log)
  print_log("Num of GPUs    : {}".format(torch.cuda.device_count()), log)
  args.dataset = args.dataset.lower()

  # Mean + Std
  if args.dataset == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif args.dataset == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
  else:
    raise TypeError("Unknow dataset : {:}".format(args.dataset))
  # Data Argumentation
  if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean, std)]
    if args.cutout > 0 : lists += [Cutout(args.cutout)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
  else:
    raise TypeError("Unknow dataset : {:}".format(args.dataset))
  # Datasets
  if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.data_path, train= True, transform=train_transform, download=True)
    test_data  = dset.CIFAR10(args.data_path, train=False, transform=test_transform , download=True)
    num_classes = 10
  elif args.dataset == 'cifar100':
    train_data = dset.CIFAR100(args.data_path, train= True, transform=train_transform, download=True)
    test_data  = dset.CIFAR100(args.data_path, train=False, transform=test_transform , download=True)
    num_classes = 100
  else:
    raise TypeError("Unknow dataset : {:}".format(args.dataset))
  # Data Loader
  if args.validate:
    indices = list(range(len(train_data)))
    split   = int(args.train_portion * len(indices))
    random.shuffle(indices)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                      pin_memory=True, num_workers=args.workers)
    test_loader  = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
                      pin_memory=True, num_workers=args.workers)
  else:
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

  # network and criterion
  criterion = torch.nn.CrossEntropyLoss().cuda()
  basemodel = Networks[args.arch](args.init_channels, num_classes, args.layers)
  model     = torch.nn.DataParallel(basemodel).cuda()
  print_log("Parameter size = {:.3f} MB".format(count_parameters_in_MB(basemodel.base_parameters())), log)
  print_log("Train-transformation : {:}\nTest--transformation : {:}".format(train_transform, test_transform), log)

  # optimizer and LR-scheduler
  base_optimizer = torch.optim.SGD (basemodel.base_parameters(), args.learning_rate_max, momentum=args.momentum, weight_decay=args.weight_decay)
  #base_optimizer = torch.optim.Adam(basemodel.base_parameters(), lr=args.learning_rate_max, betas=(0.5, 0.999), weight_decay=args.weight_decay)
  base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  arch_optimizer = torch.optim.Adam(basemodel.arch_parameters(), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  # snapshot
  checkpoint_path = os.path.join(args.save_path, 'checkpoint-search.pth')
  if args.resume is not None and os.path.isfile(args.resume):
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    basemodel.load_state_dict( checkpoint['state_dict'] )
    base_optimizer.load_state_dict( checkpoint['base_optimizer'] )
    arch_optimizer.load_state_dict( checkpoint['arch_optimizer'] )
    base_scheduler.load_state_dict( checkpoint['base_scheduler'] )
    genotypes   = checkpoint['genotypes']
    print_log('Load resume from {:} with start-epoch = {:}'.format(args.resume, start_epoch), log)
  elif os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    basemodel.load_state_dict( checkpoint['state_dict'] )
    base_optimizer.load_state_dict( checkpoint['base_optimizer'] )
    arch_optimizer.load_state_dict( checkpoint['arch_optimizer'] )
    base_scheduler.load_state_dict( checkpoint['base_scheduler'] )
    genotypes   = checkpoint['genotypes']
    print_log('Load checkpoint from {:} with start-epoch = {:}'.format(checkpoint_path, start_epoch), log)
  else:
    start_epoch, genotypes = 0, {}
    print_log('Train model-search from scratch.', log)

  config = load_config(args.model_config)

  if args.only_base:
    print_log('---- Only Train the Searched Model ----', log)
    main_procedure(config, args.dataset, args.data_path, args, basemodel.genotype(), 36, 20, log)
    return

  # Main loop
  start_time, epoch_time, total_train_time = time.time(), AverageMeter(), 0
  for epoch in range(start_epoch, args.epochs):
    base_scheduler.step()

    basemodel.set_tau( args.tau_max - epoch*1.0/args.epochs*(args.tau_max-args.tau_min) )
    #if epoch + 2 == args.epochs:
    #  torch.cuda.empty_cache()
    #  basemodel.set_gumbel(False)

    need_time = convert_secs2time(epoch_time.val * (args.epochs-epoch), True)
    print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f} ~ {:6.4f}] [Batch={:d}], tau={:}'.format(time_string(), epoch, args.epochs, need_time, min(base_scheduler.get_lr()), max(base_scheduler.get_lr()), args.batch_size, basemodel.get_tau()), log)

    genotype = basemodel.genotype()
    print_log('genotype = {:}'.format(genotype), log)

    print_log('{:03d}/{:03d} alphas :\n{:}'.format(epoch, args.epochs, return_alphas_str(basemodel)), log)

    # training
    train_acc1, train_acc5, train_obj, train_time \
                                      = train(train_loader, test_loader, model, criterion, base_optimizer, arch_optimizer, epoch, log)
    total_train_time += train_time
    # validation
    valid_acc1, valid_acc5, valid_obj = infer(test_loader, model, criterion, epoch, log)
    print_log('{:03d}/{:03d}, Train-Accuracy = {:.2f}, Test-Accuracy = {:.2f}'.format(epoch, args.epochs, train_acc1, valid_acc1), log)
    # save genotype
    genotypes[epoch] = basemodel.genotype()
    # save checkpoint
    torch.save({'epoch' : epoch + 1,
                'args'  : deepcopy(args),
                'state_dict': basemodel.state_dict(),
                'genotypes' : genotypes,
                'base_optimizer' : base_optimizer.state_dict(),
                'arch_optimizer' : arch_optimizer.state_dict(),
                'base_scheduler' : base_scheduler.state_dict()},
                checkpoint_path)
    print_log('----> Save into {:}'.format(checkpoint_path), log)


    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  print_log('Finish with training time = {:}'.format( convert_secs2time(total_train_time, True) ), log)

  # clear GPU cache
  #torch.cuda.empty_cache()
  #main_procedure(config, args.dataset, args.data_path, args, basemodel.genotype(), 36, 20, log)
  log.close()


def train(train_queue, valid_queue, model, criterion, base_optimizer, arch_optimizer, epoch, log):
  data_time, batch_time = AverageMeter(), AverageMeter()
  objs, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  model.train()

  valid_iter = iter(valid_queue)
  end = time.time()
  for step, (inputs, targets) in enumerate(train_queue):
    batch, C, H, W = inputs.size()

    #inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    data_time.update(time.time() - end)

    # get a random minibatch from the search queue with replacement
    try:
      input_search, target_search = next(valid_iter)
    except:
      valid_iter = iter(valid_queue)
      input_search, target_search = next(valid_iter)
    
    target_search = target_search.cuda(non_blocking=True)

    # update the architecture
    arch_optimizer.zero_grad()
    output_search = model(input_search)
    arch_loss = criterion(output_search, target_search)
    arch_loss.backward()
    arch_optimizer.step()

    # update the parameters
    base_optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.module.base_parameters(), args.grad_clip)
    base_optimizer.step()

    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    objs.update(loss.item() , batch)
    top1.update(prec1.item(), batch)
    top5.update(prec5.item(), batch)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % args.print_freq == 0 or (step+1) == len(train_queue):
      Sstr = ' TRAIN-SEARCH ' + time_string() + ' Epoch: [{:03d}][{:03d}/{:03d}]'.format(epoch, step, len(train_queue))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f}) Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(loss=objs, top1=top1, top5=top5)
      print_log(Sstr + ' ' + Tstr + ' ' + Lstr, log)

  return top1.avg, top5.avg, objs.avg, batch_time.sum


def infer(valid_queue, model, criterion, epoch, log):
  objs, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  
  model.eval()
  with torch.no_grad():
    for step, (inputs, targets) in enumerate(valid_queue):
      batch, C, H, W = inputs.size()
      targets = targets.cuda(non_blocking=True)

      logits = model(inputs)
      loss = criterion(logits, targets)

      prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
      objs.update(loss.item() , batch)
      top1.update(prec1.item(), batch)
      top5.update(prec5.item(), batch)

      if step % args.print_freq == 0 or (step+1) == len(valid_queue):
        Sstr = ' VALID-SEARCH ' + time_string() + ' Epoch: [{:03d}][{:03d}/{:03d}]'.format(epoch, step, len(valid_queue))
        Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f}) Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(loss=objs, top1=top1, top5=top5)
        print_log(Sstr + ' ' + Lstr, log)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 
