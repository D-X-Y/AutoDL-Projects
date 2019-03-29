import os, sys, time
from copy import deepcopy
import torch
import torch.nn as nn
import torchvision.transforms as transforms


from utils import print_log, obtain_accuracy, AverageMeter
from utils import time_string, convert_secs2time
from utils import count_parameters_in_MB
from utils import print_FLOPs
from utils import Cutout
from nas import NetworkImageNet as Network
from datasets import get_datasets


def obtain_best(accuracies):
  if len(accuracies) == 0: return (0, 0)
  tops = [value for key, value in accuracies.items()]
  s2b = sorted( tops )
  return s2b[-1]


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def main_procedure_imagenet(config, data_path, args, genotype, init_channels, layers, log):
  
  # training data and testing data
  train_data, valid_data, class_num = get_datasets('imagenet-1k', data_path, -1)

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=config.batch_size, shuffle= True, pin_memory=True, num_workers=args.workers)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

  class_num   = 1000

  print_log('-------------------------------------- main-procedure', log)
  print_log('config        : {:}'.format(config), log)
  print_log('genotype      : {:}'.format(genotype), log)
  print_log('init_channels : {:}'.format(init_channels), log)
  print_log('layers        : {:}'.format(layers), log)
  print_log('class_num     : {:}'.format(class_num), log)
  basemodel = Network(init_channels, class_num, layers, config.auxiliary, genotype)
  model     = torch.nn.DataParallel(basemodel).cuda()

  total_param, aux_param = count_parameters_in_MB(basemodel), count_parameters_in_MB(basemodel.auxiliary_param())
  print_log('Network =>\n{:}'.format(basemodel), log)
  print_FLOPs(basemodel, (1,3,224,224), [print_log, log])
  print_log('Parameters : {:} - {:} = {:.3f} MB'.format(total_param, aux_param, total_param - aux_param), log)
  print_log('config        : {:}'.format(config), log)
  print_log('genotype      : {:}'.format(genotype), log)
  print_log('Train-Dataset : {:}'.format(train_data), log)
  print_log('Valid--Dataset : {:}'.format(valid_data), log)
  print_log('Args          : {:}'.format(args), log)


  criterion = torch.nn.CrossEntropyLoss().cuda()
  criterion_smooth = CrossEntropyLabelSmooth(class_num, config.label_smooth).cuda()


  optimizer = torch.optim.SGD(model.parameters(), config.LR, momentum=config.momentum, weight_decay=config.decay, nesterov=True)
  if config.type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.epochs))
  elif config.type == 'steplr':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.decay_period, gamma=config.gamma)
  else:
    raise ValueError('Can not find the schedular type : {:}'.format(config.type))


  checkpoint_path = os.path.join(args.save_path, 'checkpoint-imagenet-model.pth')
  if os.path.isfile(checkpoint_path):
    checkpoint = torch.load( checkpoint_path )

    start_epoch = checkpoint['epoch']
    basemodel.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    accuracies  = checkpoint['accuracies']
    print_log('Load checkpoint from {:} with start-epoch = {:}'.format(checkpoint_path, start_epoch), log)
  else:
    start_epoch, accuracies = 0, {}
    print_log('Train model from scratch without pre-trained model or snapshot', log)


  # Main loop
  start_time, epoch_time = time.time(), AverageMeter()
  for epoch in range(start_epoch, config.epochs):
    scheduler.step()

    need_time = convert_secs2time(epoch_time.val * (config.epochs-epoch), True)
    print_log("\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} LR={:6.4f} ~ {:6.4f}, Batch={:d}".format(time_string(), epoch, config.epochs, need_time, min(scheduler.get_lr()), max(scheduler.get_lr()), config.batch_size), log)

    basemodel.update_drop_path(config.drop_path_prob * epoch / config.epochs)

    train_acc1, train_acc5, train_los = _train(train_queue, model, criterion_smooth, optimizer, 'train', epoch, config, args.print_freq, log)

    with torch.no_grad():
      valid_acc1, valid_acc5, valid_los = _train(valid_queue, model, criterion,           None, 'test' , epoch, config, args.print_freq, log)
    accuracies[epoch] = (valid_acc1, valid_acc5)

    torch.save({'epoch'     : epoch + 1,
                'args'      : deepcopy(args),
                'state_dict': basemodel.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'accuracies': accuracies},
                checkpoint_path)
    best_acc = obtain_best( accuracies )
    print_log('----> Best Accuracy : Acc@1={:.2f}, Acc@5={:.2f}, Error@1={:.2f}, Error@5={:.2f}'.format(best_acc[0], best_acc[1], 100-best_acc[0], 100-best_acc[1]), log)
    print_log('----> Save into {:}'.format(checkpoint_path), log)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()


def _train(xloader, model, criterion, optimizer, mode, epoch, config, print_freq, log):
  data_time, batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  if mode == 'train':
    model.train()
  elif mode == 'test':
    model.eval()
  else: raise ValueError("The mode is not right : {:}".format(mode))
  
  end = time.time()
  for i, (inputs, targets) in enumerate(xloader):
    # measure data loading time
    data_time.update(time.time() - end)
    # calculate prediction and loss
    targets = targets.cuda(non_blocking=True)

    if mode == 'train': optimizer.zero_grad()

    if config.auxiliary and model.training:
      logits, logits_aux = model(inputs)
    else:
      logits = model(inputs)

    loss = criterion(logits, targets)
    if config.auxiliary and model.training:
      loss_aux = criterion(logits_aux, targets)
      loss += config.auxiliary_weight * loss_aux
    
    if mode == 'train':
      loss.backward()
      if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
      optimizer.step()
    # record
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % print_freq == 0 or (i+1) == len(xloader):
      Sstr = ' {:5s}'.format(mode) + time_string() + ' Epoch: [{:03d}][{:03d}/{:03d}]'.format(epoch, i, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(loss=losses, top1=top1, top5=top5)
      print_log(Sstr + ' ' + Tstr + ' ' + Lstr, log)

  print_log ('{TIME:} **{mode:}** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Loss:{loss:.3f}'.format(TIME=time_string(), mode=mode, top1=top1, top5=top5, error1=100-top1.avg, error5=100-top5.avg, loss=losses.avg), log)
  return top1.avg, top5.avg, losses.avg
