##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, time, torch
from log_utils import AverageMeter, time_string
from utils     import obtain_accuracy
from models    import change_key


def get_flop_loss(expected_flop, flop_cur, flop_need, flop_tolerant):
  expected_flop = torch.mean( expected_flop )

  if flop_cur < flop_need - flop_tolerant:   # Too Small FLOP
    loss = - torch.log( expected_flop )
  #elif flop_cur > flop_need + flop_tolerant: # Too Large FLOP
  elif flop_cur > flop_need: # Too Large FLOP
    loss = torch.log( expected_flop )
  else: # Required FLOP
    loss = None
  if loss is None: return 0, 0
  else           : return loss, loss.item()


def search_train(search_loader, network, criterion, scheduler, base_optimizer, arch_optimizer, optim_config, extra_info, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, arch_losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  arch_cls_losses, arch_flop_losses = AverageMeter(), AverageMeter()
  epoch_str, flop_need, flop_weight, flop_tolerant = extra_info['epoch-str'], extra_info['FLOP-exp'], extra_info['FLOP-weight'], extra_info['FLOP-tolerant']

  network.train()
  logger.log('[Search] : {:}, FLOP-Require={:.2f} MB, FLOP-WEIGHT={:.2f}'.format(epoch_str, flop_need, flop_weight))
  end = time.time()
  network.apply( change_key('search_mode', 'search') )
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(search_loader):
    scheduler.update(None, 1.0 * step / len(search_loader))
    # calculate prediction and loss
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    base_optimizer.zero_grad()
    logits, expected_flop = network(base_inputs)
    #network.apply( change_key('search_mode', 'basic') )
    #features, logits = network(base_inputs)
    base_loss = criterion(logits, base_targets)
    base_loss.backward()
    base_optimizer.step()
    # record
    prec1, prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(), base_inputs.size(0))
    top1.update       (prec1.item(), base_inputs.size(0))
    top5.update       (prec5.item(), base_inputs.size(0))

    # update the architecture
    arch_optimizer.zero_grad()
    logits, expected_flop = network(arch_inputs)
    flop_cur  = network.module.get_flop('genotype', None, None)
    flop_loss, flop_loss_scale = get_flop_loss(expected_flop, flop_cur, flop_need, flop_tolerant)
    acls_loss = criterion(logits, arch_targets)
    arch_loss = acls_loss + flop_loss * flop_weight
    arch_loss.backward()
    arch_optimizer.step()
  
    # record
    arch_losses.update(arch_loss.item(), arch_inputs.size(0))
    arch_flop_losses.update(flop_loss_scale, arch_inputs.size(0))
    arch_cls_losses.update (acls_loss.item(), arch_inputs.size(0))
    
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    if step % print_freq == 0 or (step+1) == len(search_loader):
      Sstr = '**TRAIN** ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(search_loader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Lstr = 'Base-Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(loss=base_losses, top1=top1, top5=top5)
      Vstr = 'Acls-loss {aloss.val:.3f} ({aloss.avg:.3f}) FLOP-Loss {floss.val:.3f} ({floss.avg:.3f}) Arch-Loss {loss.val:.3f} ({loss.avg:.3f})'.format(aloss=arch_cls_losses, floss=arch_flop_losses, loss=arch_losses)
      logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Vstr)
      #Istr = 'Bsz={:} Asz={:}'.format(list(base_inputs.size()), list(arch_inputs.size()))
      #logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Vstr + ' ' + Istr)
      #print(network.module.get_arch_info())
      #print(network.module.width_attentions[0])
      #print(network.module.width_attentions[1])

  logger.log(' **TRAIN** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Base-Loss:{baseloss:.3f}, Arch-Loss={archloss:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg, error5=100-top5.avg, baseloss=base_losses.avg, archloss=arch_losses.avg))
  return base_losses.avg, arch_losses.avg, top1.avg, top5.avg



def search_valid(xloader, network, criterion, extra_info, print_freq, logger):
  data_time, batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

  network.eval()
  network.apply( change_key('search_mode', 'search') )
  end = time.time()
  #logger.log('Starting evaluating {:}'.format(epoch_info))
  with torch.no_grad():
    for i, (inputs, targets) in enumerate(xloader):
      # measure data loading time
      data_time.update(time.time() - end)
      # calculate prediction and loss
      targets = targets.cuda(non_blocking=True)

      logits, expected_flop = network(inputs)
      loss             = criterion(logits, targets)
      # record
      prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
      losses.update(loss.item(),  inputs.size(0))
      top1.update  (prec1.item(), inputs.size(0))
      top5.update  (prec5.item(), inputs.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % print_freq == 0 or (i+1) == len(xloader):
        Sstr = '**VALID** ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(extra_info, i, len(xloader))
        Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
        Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(loss=losses, top1=top1, top5=top5)
        Istr = 'Size={:}'.format(list(inputs.size()))
        logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Istr)

  logger.log(' **VALID** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Loss:{loss:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg, error5=100-top5.avg, loss=losses.avg))
 
  return losses.avg, top1.avg, top5.avg
