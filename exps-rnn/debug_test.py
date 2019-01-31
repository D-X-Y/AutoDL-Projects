import os, gc, sys, time, math
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from utils import print_log, obtain_accuracy, AverageMeter
from utils import time_string, convert_secs2time
from utils import count_parameters_in_MB
from datasets import Corpus
from nas_rnn import batchify, get_batch, repackage_hidden
from nas_rnn import DARTS
from nas_rnn import DARTSCell, RNNModel
from nas_rnn import basemodel as model
from scheduler import load_config


def main_procedure(config, genotype, print_freq, log):
 
  print_log('-'*90, log)
  print_log('genotype : {:}'.format(genotype), log)
  print_log('config   : {:}'.format(config.bptt), log)

  corpus = Corpus(config.data_path)
  train_data = batchify(corpus.train, config.train_batch, True)
  valid_data = batchify(corpus.valid, config.eval_batch , True)
  test_data  = batchify(corpus.test,  config.test_batch , True)
  ntokens = len(corpus.dictionary)
  print_log("Train--Data Size : {:}".format(train_data.size()), log)
  print_log("Valid--Data Size : {:}".format(valid_data.size()), log)
  print_log("Test---Data Size : {:}".format( test_data.size()), log)
  print_log("ntokens = {:}".format(ntokens), log)

  model = RNNModel(ntokens, config.emsize, config.nhid, config.nhidlast, 
                       config.dropout, config.dropouth, config.dropoutx, config.dropouti, config.dropoute, 
                       cell_cls=DARTSCell, genotype=genotype)
  model = model.cuda()
  print_log('Network =>\n{:}'.format(model), log)
  print_log('Genotype : {:}'.format(genotype), log)
  print_log('Parameters : {:.3f} MB'.format(count_parameters_in_MB(model)), log)


  print_log('--------------------- Finish Training ----------------', log)
  test_loss = evaluate(model, corpus, test_data , config.test_batch, config.bptt)
  print_log('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)), log)
  vali_loss = evaluate(model, corpus, valid_data, config.eval_batch, config.bptt)
  print_log('| End of training | valid loss {:5.2f} | valid ppl {:8.2f}'.format(vali_loss, math.exp(vali_loss)), log)
  


def evaluate(model, corpus, data_source, batch_size, bptt):
  # Turn on evaluation mode which disables dropout.
  model.eval()
  total_loss, total_length = 0.0, 0.0
  with torch.no_grad():
    ntokens = len(corpus.dictionary)
    hidden  = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, bptt):
      data, targets = get_batch(data_source, i, bptt)
      targets = targets.view(-1)

      log_prob, hidden = model(data, hidden)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)

      total_loss += loss.item() * len(data)
      total_length += len(data)
      hidden = repackage_hidden(hidden)
  return total_loss / total_length

if __name__ == '__main__':
  path = './configs/NAS-PTB-BASE.config'
  config = load_config(path)
  main_procedure(config, DARTS, 10, None)
