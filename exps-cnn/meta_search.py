import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import TieredImageNet, MetaBatchSampler
from utils import AverageMeter, time_string, convert_secs2time
from utils import print_log, obtain_accuracy
from utils import Cutout, count_parameters_in_MB
from meta_nas import return_alphas_str, MetaNetwork
from train_utils import main_procedure
from scheduler import load_config

Networks = {'meta': MetaNetwork}

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data_path',          type=str,   help='Path to dataset')
parser.add_argument('--arch',               type=str,   choices=Networks.keys(), help='Choose networks.')
parser.add_argument('--n_way',              type=int,   help='N-WAY.')
parser.add_argument('--k_shot',             type=int,   help='K-SHOT.')
# Learning Parameters
parser.add_argument('--learning_rate_max',  type=float, help='initial learning rate')
parser.add_argument('--learning_rate_min',  type=float, help='minimum learning rate')
parser.add_argument('--momentum',           type=float, help='momentum')
parser.add_argument('--weight_decay',       type=float, help='weight decay')
parser.add_argument('--epochs',             type=int,   help='num of training epochs')
# architecture leraning rate
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
#
parser.add_argument('--init_channels',      type=int, help='num of init channels')
parser.add_argument('--layers',             type=int, help='total number of layers')
# 
parser.add_argument('--cutout',             type=int,   help='cutout length, negative means no cutout')
parser.add_argument('--grad_clip',          type=float, help='gradient clipping')
parser.add_argument('--model_config',       type=str  , help='the model configuration')

# resume
parser.add_argument('--resume',             type=str  , help='the resume path')
parser.add_argument('--only_base',action='store_true', default=False, help='only train the searched model')
# split data
parser.add_argument('--validate', action='store_true', default=False, help='split train-data int train/val or not')
parser.add_argument('--train_portion',      type=float, default=0.5, help='portion of training data')
# log
parser.add_argument('--workers',            type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--save_path',          type=str, help='Folder to save checkpoints and log.')
parser.add_argument('--print_freq',         type=int, help='print frequency (default: 200)')
parser.add_argument('--manualSeed',         type=int, help='manual seed')
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

  # Mean + Std
  means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  # Data Argumentation
  lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(80, padding=4), transforms.ToTensor(),
           transforms.Normalize(means, stds)]
  if args.cutout > 0 : lists += [Cutout(args.cutout)]
  train_transform = transforms.Compose(lists)
  test_transform  = transforms.Compose([transforms.CenterCrop(80), transforms.ToTensor(), transforms.Normalize(means, stds)])
  
  train_data = TieredImageNet(args.data_path, 'train', train_transform)
  test_data  = TieredImageNet(args.data_path, 'val'  , test_transform )

  train_sampler = MetaBatchSampler(train_data.labels, args.n_way, args.k_shot * 2, len(train_data) // (args.n_way*args.k_shot))
  test_sampler  = MetaBatchSampler( test_data.labels, args.n_way, args.k_shot * 2, len( test_data) // (args.n_way*args.k_shot))

  train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=train_sampler)
  test_loader  = torch.utils.data.DataLoader( test_data, batch_sampler= test_sampler)

  # network
  basemodel = Networks[args.arch](args.init_channels, args.layers, head='imagenet')
  model     = torch.nn.DataParallel(basemodel).cuda()
  print_log("Parameter size = {:.3f} MB".format(count_parameters_in_MB(basemodel.base_parameters())), log)
  print_log("Train-transformation : {:}\nTest--transformation : {:}".format(train_transform, test_transform), log)

  # optimizer and LR-scheduler
  #base_optimizer = torch.optim.SGD (basemodel.base_parameters(), args.learning_rate_max, momentum=args.momentum, weight_decay=args.weight_decay)
  base_optimizer = torch.optim.Adam(basemodel.base_parameters(), lr=args.learning_rate_max, betas=(0.5, 0.999), weight_decay=args.weight_decay)
  base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  arch_optimizer = torch.optim.Adam(basemodel.arch_parameters(), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  # snapshot
  checkpoint_path = os.path.join(args.save_path, 'checkpoint-meta-search.pth')
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
    CIFAR_DATA_DIR = os.environ['TORCH_HOME'] + '/cifar.python'
    main_procedure(config, 'cifar10', CIFAR_DATA_DIR, args, basemodel.genotype(), 36, 20, log)
    return

  # Main loop
  start_time, epoch_time, total_train_time = time.time(), AverageMeter(), 0
  for epoch in range(start_epoch, args.epochs):
    base_scheduler.step()

    need_time = convert_secs2time(epoch_time.val * (args.epochs-epoch), True)
    print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f} ~ {:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, min(base_scheduler.get_lr()), max(base_scheduler.get_lr())), log)

    genotype = basemodel.genotype()
    print_log('genotype = {:}'.format(genotype), log)
    print_log('{:03d}/{:03d} alphas :\n{:}'.format(epoch, args.epochs, return_alphas_str(basemodel)), log)

    # training
    train_acc1, train_obj, train_time \
                                      = train(train_loader, test_loader, model, args.n_way, base_optimizer, arch_optimizer, epoch, log)
    total_train_time += train_time
    # validation
    valid_acc1, valid_obj = infer(test_loader, model, epoch, args.n_way, log)

    print_log('META -> {:}-way {:}-shot : {:03d}/{:03d} : Train Acc : {:.2f}, Test Acc : {:.2f}'.format(args.n_way, args.k_shot, epoch, args.epochs, train_acc1, valid_acc1), log)
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
  CIFAR_DATA_DIR = os.environ['TORCH_HOME'] + '/cifar.python'
  print_log('test for CIFAR-10', log)
  torch.cuda.empty_cache()
  main_procedure(config, 'cifar10' , CIFAR_DATA_DIR, args, basemodel.genotype(), 36, 20, log)
  print_log('test for CIFAR-100', log)
  torch.cuda.empty_cache()
  main_procedure(config, 'cifar100', CIFAR_DATA_DIR, args, basemodel.genotype(), 36, 20, log)
  log.close()



def euclidean_dist(A, B):
  na, da = A.size()
  nb, db = B.size()
  assert da == db, 'invalid feature dim : {:} vs. {:}'.format(da, db)
  X, Y = A.view(na, 1, da), B.view(1, nb, db)
  return torch.pow(X-Y, 2).sum(2)
  


def get_loss(features, targets, n_way):
  classes = torch.unique(targets)
  shot = features.size(0) // n_way // 2

  support_index, query_index, labels = [], [], []
  for idx, cls in enumerate( classes.tolist() ):
    indexs = (targets == cls).nonzero().view(-1).tolist()
    support_index.append(indexs[:shot])
    query_index   += indexs[shot:]
    labels        += [idx] * shot
  query_features = features[query_index, :]
  support_features = features[support_index, :]
  support_features = torch.mean(support_features, dim=1)
    
  labels = torch.LongTensor(labels).cuda(non_blocking=True)
  logits = -euclidean_dist(query_features, support_features)
  loss = F.cross_entropy(logits, labels)
  accuracy = obtain_accuracy(logits.data, labels.data, topk=(1,))[0]
  return loss, accuracy



def train(train_queue, valid_queue, model, n_way, base_optimizer, arch_optimizer, epoch, log):
  data_time, batch_time = AverageMeter(), AverageMeter()
  objs, accuracies = AverageMeter(), AverageMeter()
  model.train()

  valid_iter = iter(valid_queue)
  end = time.time()
  for step, (inputs, targets) in enumerate(train_queue):
    batch, C, H, W = inputs.size()

    #inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
    #targets = targets.cuda(non_blocking=True)
    data_time.update(time.time() - end)

    # get a random minibatch from the search queue with replacement
    try:
      input_search, target_search = next(valid_iter)
    except:
      valid_iter = iter(valid_queue)
      input_search, target_search = next(valid_iter)
    
    #target_search = target_search.cuda(non_blocking=True)

    # update the architecture
    arch_optimizer.zero_grad()
    feature_search = model(input_search)
    arch_loss, arch_accuracy = get_loss(feature_search, target_search, n_way)
    arch_loss.backward()
    arch_optimizer.step()

    # update the parameters
    base_optimizer.zero_grad()
    feature_model = model(inputs)
    model_loss, model_accuracy = get_loss(feature_model, targets, n_way)

    model_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.module.base_parameters(), args.grad_clip)
    base_optimizer.step()

    objs.update(model_loss.item() , batch)
    accuracies.update(model_accuracy.item(), batch)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % args.print_freq == 0 or (step+1) == len(train_queue):
      Sstr = ' TRAIN-SEARCH ' + time_string() + ' Epoch: [{:03d}][{:03d}/{:03d}]'.format(epoch, step, len(train_queue))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f}) Prec@1 {top1.val:.2f} ({top1.avg:.2f})'.format(loss=objs, top1=accuracies)
      Istr = 'I : {:}'.format( list(inputs.size()) )
      print_log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Istr, log)

  return accuracies.avg, objs.avg, batch_time.sum



def infer(valid_queue, model, epoch, n_way, log):
  objs, accuracies = AverageMeter(), AverageMeter()
  
  model.eval()
  with torch.no_grad():
    for step, (inputs, targets) in enumerate(valid_queue):
      batch, C, H, W = inputs.size()
      #targets = targets.cuda(non_blocking=True)

      features = model(inputs)
      loss, accuracy = get_loss(features, targets, n_way)

      objs.update(loss.item() , batch)
      accuracies.update(accuracy.item(), batch)

      if step % (args.print_freq*4) == 0 or (step+1) == len(valid_queue):
        Sstr = ' VALID-SEARCH ' + time_string() + ' Epoch: [{:03d}][{:03d}/{:03d}]'.format(epoch, step, len(valid_queue))
        Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f}) Prec@1 {top1.val:.2f} ({top1.avg:.2f})'.format(loss=objs, top1=accuracies)
        print_log(Sstr + ' ' + Lstr, log)

  return accuracies.avg, objs.avg



if __name__ == '__main__':
  main() 
