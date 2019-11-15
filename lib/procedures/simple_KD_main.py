##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, time, torch
import torch.nn.functional as F
# our modules
from log_utils import AverageMeter, time_string
from utils     import obtain_accuracy


def simple_KD_train(xloader, teacher, network, criterion, scheduler, optimizer, optim_config, extra_info, print_freq, logger):
  loss, acc1, acc5 = procedure(xloader, teacher, network, criterion, scheduler, optimizer, 'train', optim_config, extra_info, print_freq, logger)
  return loss, acc1, acc5

def simple_KD_valid(xloader, teacher, network, criterion, optim_config, extra_info, print_freq, logger):
  with torch.no_grad():
    loss, acc1, acc5 = procedure(xloader, teacher, network, criterion, None, None, 'valid', optim_config, extra_info, print_freq, logger)
  return loss, acc1, acc5


def loss_KD_fn(criterion, student_logits, teacher_logits, studentFeatures, teacherFeatures, targets, alpha, temperature):
  basic_loss = criterion(student_logits, targets) * (1. - alpha)
  log_student= F.log_softmax(student_logits / temperature, dim=1)
  sof_teacher= F.softmax    (teacher_logits / temperature, dim=1)
  KD_loss    = F.kl_div(log_student, sof_teacher, reduction='batchmean') * (alpha * temperature * temperature)
  return basic_loss + KD_loss


def procedure(xloader, teacher, network, criterion, scheduler, optimizer, mode, config, extra_info, print_freq, logger):
  data_time, batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  Ttop1, Ttop5 = AverageMeter(), AverageMeter()
  if mode == 'train':
    network.train()
  elif mode == 'valid':
    network.eval()
  else: raise ValueError("The mode is not right : {:}".format(mode))
  teacher.eval()
  
  logger.log('[{:5s}] config :: auxiliary={:}, KD :: [alpha={:.2f}, temperature={:.2f}]'.format(mode, config.auxiliary if hasattr(config, 'auxiliary') else -1, config.KD_alpha, config.KD_temperature))
  end = time.time()
  for i, (inputs, targets) in enumerate(xloader):
    if mode == 'train': scheduler.update(None, 1.0 * i / len(xloader))
    # measure data loading time
    data_time.update(time.time() - end)
    # calculate prediction and loss
    targets = targets.cuda(non_blocking=True)

    if mode == 'train': optimizer.zero_grad()

    student_f, logits = network(inputs)
    if isinstance(logits, list):
      assert len(logits) == 2, 'logits must has {:} items instead of {:}'.format(2, len(logits))
      logits, logits_aux = logits
    else:
      logits, logits_aux = logits, None
    with torch.no_grad():
      teacher_f, teacher_logits = teacher(inputs)

    loss             = loss_KD_fn(criterion, logits, teacher_logits, student_f, teacher_f, targets, config.KD_alpha, config.KD_temperature)
    if config is not None and hasattr(config, 'auxiliary') and config.auxiliary > 0:
      loss_aux = criterion(logits_aux, targets)
      loss += config.auxiliary * loss_aux
    
    if mode == 'train':
      loss.backward()
      optimizer.step()

    # record
    sprec1, sprec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),   inputs.size(0))
    top1.update  (sprec1.item(), inputs.size(0))
    top5.update  (sprec5.item(), inputs.size(0))
    # teacher
    tprec1, tprec5 = obtain_accuracy(teacher_logits.data, targets.data, topk=(1, 5))
    Ttop1.update (tprec1.item(), inputs.size(0))
    Ttop5.update (tprec5.item(), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % print_freq == 0 or (i+1) == len(xloader):
      Sstr = ' {:5s} '.format(mode.upper()) + time_string() + ' [{:}][{:03d}/{:03d}]'.format(extra_info, i, len(xloader))
      if scheduler is not None:
        Sstr += ' {:}'.format(scheduler.get_min_info())
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(loss=losses, top1=top1, top5=top5)
      Lstr+= ' Teacher : acc@1={:.2f}, acc@5={:.2f}'.format(Ttop1.avg, Ttop5.avg)
      Istr = 'Size={:}'.format(list(inputs.size()))
      logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Istr)

  logger.log(' **{:5s}** accuracy drop :: @1={:.2f}, @5={:.2f}'.format(mode.upper(), Ttop1.avg - top1.avg, Ttop5.avg - top5.avg))
  logger.log(' **{mode:5s}** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Loss:{loss:.3f}'.format(mode=mode.upper(), top1=top1, top5=top5, error1=100-top1.avg, error5=100-top5.avg, loss=losses.avg))
  return losses.avg, top1.avg, top5.avg
