#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import math, torch
import torch.nn as nn
from bisect import bisect_right
from torch.optim import Optimizer


class _LRScheduler(object):

  def __init__(self, optimizer, warmup_epochs, epochs):
    if not isinstance(optimizer, Optimizer):
      raise TypeError('{:} is not an Optimizer'.format(type(optimizer).__name__))
    self.optimizer = optimizer
    for group in optimizer.param_groups:
      group.setdefault('initial_lr', group['lr'])
    self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
    self.max_epochs = epochs
    self.warmup_epochs  = warmup_epochs
    self.current_epoch  = 0
    self.current_iter   = 0

  def extra_repr(self):
    return ''

  def __repr__(self):
    return ('{name}(warmup={warmup_epochs}, max-epoch={max_epochs}, current::epoch={current_epoch}, iter={current_iter:.2f}'.format(name=self.__class__.__name__, **self.__dict__)
              + ', {:})'.format(self.extra_repr()))

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)

  def get_lr(self):
    raise NotImplementedError

  def get_min_info(self):
    lrs = self.get_lr()
    return '#LR=[{:.6f}~{:.6f}] epoch={:03d}, iter={:4.2f}#'.format(min(lrs), max(lrs), self.current_epoch, self.current_iter)

  def get_min_lr(self):
    return min( self.get_lr() )

  def update(self, cur_epoch, cur_iter):
    if cur_epoch is not None:
      assert isinstance(cur_epoch, int) and cur_epoch>=0, 'invalid cur-epoch : {:}'.format(cur_epoch)
      self.current_epoch = cur_epoch
    if cur_iter is not None:
      assert isinstance(cur_iter, float) and cur_iter>=0, 'invalid cur-iter : {:}'.format(cur_iter)
      self.current_iter  = cur_iter
    for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
      param_group['lr'] = lr



class CosineAnnealingLR(_LRScheduler):

  def __init__(self, optimizer, warmup_epochs, epochs, T_max, eta_min):
    self.T_max = T_max
    self.eta_min = eta_min
    super(CosineAnnealingLR, self).__init__(optimizer, warmup_epochs, epochs)

  def extra_repr(self):
    return 'type={:}, T-max={:}, eta-min={:}'.format('cosine', self.T_max, self.eta_min)

  def get_lr(self):
    lrs = []
    for base_lr in self.base_lrs:
      if self.current_epoch >= self.warmup_epochs and self.current_epoch < self.max_epochs:
        last_epoch = self.current_epoch - self.warmup_epochs
        #if last_epoch < self.T_max:
        #if last_epoch < self.max_epochs:
        lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * last_epoch / self.T_max)) / 2
        #else:
        #  lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.T_max-1.0) / self.T_max)) / 2
      elif self.current_epoch >= self.max_epochs:
        lr = self.eta_min
      else:
        lr = (self.current_epoch / self.warmup_epochs + self.current_iter / self.warmup_epochs) * base_lr
      lrs.append( lr )
    return lrs



class MultiStepLR(_LRScheduler):

  def __init__(self, optimizer, warmup_epochs, epochs, milestones, gammas):
    assert len(milestones) == len(gammas), 'invalid {:} vs {:}'.format(len(milestones), len(gammas))
    self.milestones = milestones
    self.gammas     = gammas
    super(MultiStepLR, self).__init__(optimizer, warmup_epochs, epochs)

  def extra_repr(self):
    return 'type={:}, milestones={:}, gammas={:}, base-lrs={:}'.format('multistep', self.milestones, self.gammas, self.base_lrs)

  def get_lr(self):
    lrs = []
    for base_lr in self.base_lrs:
      if self.current_epoch >= self.warmup_epochs:
        last_epoch = self.current_epoch - self.warmup_epochs
        idx = bisect_right(self.milestones, last_epoch)
        lr = base_lr
        for x in self.gammas[:idx]: lr *= x
      else:
        lr = (self.current_epoch / self.warmup_epochs + self.current_iter / self.warmup_epochs) * base_lr
      lrs.append( lr )
    return lrs


class ExponentialLR(_LRScheduler):

  def __init__(self, optimizer, warmup_epochs, epochs, gamma):
    self.gamma      = gamma
    super(ExponentialLR, self).__init__(optimizer, warmup_epochs, epochs)

  def extra_repr(self):
    return 'type={:}, gamma={:}, base-lrs={:}'.format('exponential', self.gamma, self.base_lrs)

  def get_lr(self):
    lrs = []
    for base_lr in self.base_lrs:
      if self.current_epoch >= self.warmup_epochs:
        last_epoch = self.current_epoch - self.warmup_epochs
        assert last_epoch >= 0, 'invalid last_epoch : {:}'.format(last_epoch)
        lr = base_lr * (self.gamma ** last_epoch)
      else:
        lr = (self.current_epoch / self.warmup_epochs + self.current_iter / self.warmup_epochs) * base_lr
      lrs.append( lr )
    return lrs


class LinearLR(_LRScheduler):

  def __init__(self, optimizer, warmup_epochs, epochs, max_LR, min_LR):
    self.max_LR = max_LR
    self.min_LR = min_LR
    super(LinearLR, self).__init__(optimizer, warmup_epochs, epochs)

  def extra_repr(self):
    return 'type={:}, max_LR={:}, min_LR={:}, base-lrs={:}'.format('LinearLR', self.max_LR, self.min_LR, self.base_lrs)

  def get_lr(self):
    lrs = []
    for base_lr in self.base_lrs:
      if self.current_epoch >= self.warmup_epochs:
        last_epoch = self.current_epoch - self.warmup_epochs
        assert last_epoch >= 0, 'invalid last_epoch : {:}'.format(last_epoch)
        ratio = (self.max_LR - self.min_LR) * last_epoch / self.max_epochs / self.max_LR
        lr = base_lr * (1-ratio)
      else:
        lr = (self.current_epoch / self.warmup_epochs + self.current_iter / self.warmup_epochs) * base_lr
      lrs.append( lr )
    return lrs



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



def get_optim_scheduler(parameters, config):
  assert hasattr(config, 'optim') and hasattr(config, 'scheduler') and hasattr(config, 'criterion'), 'config must have optim / scheduler / criterion keys instead of {:}'.format(config)
  if config.optim == 'SGD':
    optim = torch.optim.SGD(parameters, config.LR, momentum=config.momentum, weight_decay=config.decay, nesterov=config.nesterov)
  elif config.optim == 'RMSprop':
    optim = torch.optim.RMSprop(parameters, config.LR, momentum=config.momentum, weight_decay=config.decay)
  else:
    raise ValueError('invalid optim : {:}'.format(config.optim))

  if config.scheduler == 'cos':
    T_max = getattr(config, 'T_max', config.epochs)
    scheduler = CosineAnnealingLR(optim, config.warmup, config.epochs, T_max, config.eta_min)
  elif config.scheduler == 'multistep':
    scheduler = MultiStepLR(optim, config.warmup, config.epochs, config.milestones, config.gammas)
  elif config.scheduler == 'exponential':
    scheduler = ExponentialLR(optim, config.warmup, config.epochs, config.gamma)
  elif config.scheduler == 'linear':
    scheduler = LinearLR(optim, config.warmup, config.epochs, config.LR, config.LR_min)
  else:
    raise ValueError('invalid scheduler : {:}'.format(config.scheduler))

  if config.criterion == 'Softmax':
    criterion = torch.nn.CrossEntropyLoss()
  elif config.criterion == 'SmoothSoftmax':
    criterion = CrossEntropyLabelSmooth(config.class_num, config.label_smooth)
  else:
    raise ValueError('invalid criterion : {:}'.format(config.criterion))
  return optim, scheduler, criterion
