# This file is for experimental usage
import os, sys, torch, random
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch.nn as nn

from utils  import obtain_accuracy
from models import CellStructure
from log_utils import time_string

def evaluate_one_shot(model, xloader, api, cal_mode, seed=111):
  weights = deepcopy(model.state_dict())
  model.train(cal_mode)
  with torch.no_grad():
    logits = nn.functional.log_softmax(model.arch_parameters, dim=-1)
    archs = CellStructure.gen_all(model.op_names, model.max_nodes, False)
    probs, accuracies, gt_accs = [], [], []
    loader_iter = iter(xloader)
    random.seed(seed)
    random.shuffle(archs)
    for idx, arch in enumerate(archs):
      arch_index = api.query_index_by_arch( arch )
      metrics = api.get_more_info(arch_index, 'cifar10-valid', None, False, False)
      gt_accs.append( metrics['valid-accuracy'] )
      select_logits = []
      for i, node_info in enumerate(arch.nodes):
        for op, xin in node_info:
          node_str = '{:}<-{:}'.format(i+1, xin)
          op_index = model.op_names.index(op)
          select_logits.append( logits[model.edge2index[node_str], op_index] )
      cur_prob = sum(select_logits).item()
      probs.append( cur_prob )
    cor_prob = np.corrcoef(probs, gt_accs)[0,1]
    print ('correlation for probabilities : {:}'.format(cor_prob))
      
    for idx, arch in enumerate(archs):
      model.set_cal_mode('dynamic', arch)
      try:
        inputs, targets = next(loader_iter)
      except:
        loader_iter = iter(xloader)
        inputs, targets = next(loader_iter)
      _, logits = model(inputs.cuda())
      _, preds  = torch.max(logits, dim=-1)
      correct = (preds == targets.cuda() ).float()
      accuracies.append( correct.mean().item() )
      if idx != 0 and (idx % 300 == 0 or idx + 1 == len(archs) or idx == 10):
        cor_accs = np.corrcoef(accuracies, gt_accs[:idx+1])[0,1]
        print ('{:} {:03d}/{:03d} mode={:5s}, correlation : accs={:.4f}, arch={:}'.format(time_string(), idx, len(archs), 'Train' if cal_mode else 'Eval', cor_accs, arch))
  model.load_state_dict(weights)
  return archs, probs, accuracies
