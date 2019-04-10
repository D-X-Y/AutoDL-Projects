##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
from bisect import bisect_right


class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):

  def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
    if not list(milestones) == sorted(milestones):
      raise ValueError('Milestones should be a list of'
                       ' increasing integers. Got {:}', milestones)
    assert len(milestones) == len(gammas), '{:} vs {:}'.format(milestones, gammas)
    self.milestones = milestones
    self.gammas = gammas
    super(MultiStepLR, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    LR = 1
    for x in self.gammas[:bisect_right(self.milestones, self.last_epoch)]: LR = LR * x
    return [base_lr * LR for base_lr in self.base_lrs]


def obtain_scheduler(config, optimizer):
  if config.type == 'multistep':
    scheduler = MultiStepLR(optimizer, milestones=config.milestones, gammas=config.gammas)
  elif config.type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
  else:
    raise ValueError('Unknown learning rate scheduler type : {:}'.format(config.type))
  return scheduler
