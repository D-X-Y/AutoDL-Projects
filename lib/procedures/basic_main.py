##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, time, torch
from log_utils import AverageMeter, time_string
from utils     import obtain_accuracy


def basic_train(xloader, network, criterion, scheduler, optimizer, optim_config, extra_info, print_freq, logger):
  loss, acc1, acc5 = procedure(xloader, network, criterion, scheduler, optimizer, 'train', optim_config, extra_info, print_freq, logger)
  return loss, acc1, acc5


def basic_valid(xloader, network, criterion, optim_config, extra_info, print_freq, logger):
  with torch.no_grad():
    loss, acc1, acc5 = procedure(xloader, network, criterion, None, None, 'valid', None, extra_info, print_freq, logger)
  return loss, acc1, acc5


def procedure(xloader, network, criterion, scheduler, optimizer, mode, config, extra_info, print_freq, logger):
  data_time, batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  if mode == 'train':
    network.train()
  elif mode == 'valid':
    network.eval()
  else: raise ValueError("The mode is not right : {:}".format(mode))
  
  #logger.log('[{:5s}] config ::  auxiliary={:}, message={:}'.format(mode, config.auxiliary if hasattr(config, 'auxiliary') else -1, network.module.get_message()))
  logger.log('[{:5s}] config ::  auxiliary={:}'.format(mode, config.auxiliary if hasattr(config, 'auxiliary') else -1))
  end = time.time()
  for i, (inputs, targets) in enumerate(xloader):
    if mode == 'train': scheduler.update(None, 1.0 * i / len(xloader))
    # measure data loading time
    data_time.update(time.time() - end)
    # calculate prediction and loss
    targets = targets.cuda(non_blocking=True)

    if mode == 'train': optimizer.zero_grad()

    features, logits = network(inputs)
    if isinstance(logits, list):
      assert len(logits) == 2, 'logits must has {:} items instead of {:}'.format(2, len(logits))
      logits, logits_aux = logits
    else:
      logits, logits_aux = logits, None
    loss             = criterion(logits, targets)
    if config is not None and hasattr(config, 'auxiliary') and config.auxiliary > 0:
      loss_aux = criterion(logits_aux, targets)
      loss += config.auxiliary * loss_aux
    
    if mode == 'train':
      loss.backward()
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
      Sstr = ' {:5s} '.format(mode.upper()) + time_string() + ' [{:}][{:03d}/{:03d}]'.format(extra_info, i, len(xloader))
      if scheduler is not None:
        Sstr += ' {:}'.format(scheduler.get_min_info())
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(loss=losses, top1=top1, top5=top5)
      Istr = 'Size={:}'.format(list(inputs.size()))
      logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Istr)

  logger.log(' **{mode:5s}** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Loss:{loss:.3f}'.format(mode=mode.upper(), top1=top1, top5=top5, error1=100-top1.avg, error5=100-top5.avg, loss=losses.avg))
  return losses.avg, top1.avg, top5.avg
