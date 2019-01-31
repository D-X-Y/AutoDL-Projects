import os, gc, sys, time, math
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from utils import print_log, obtain_accuracy, AverageMeter
from utils import time_string, convert_secs2time
from utils import count_parameters_in_MB
from datasets import Corpus
from nas_rnn import batchify, get_batch, repackage_hidden
from nas_rnn import DARTSCell, RNNModel


def obtain_best(accuracies):
  if len(accuracies) == 0: return (0, 0)
  tops = [value for key, value in accuracies.items()]
  s2b = sorted( tops )
  return s2b[-1]


def main_procedure(config, genotype, save_dir, print_freq, log):
 
  print_log('-'*90, log)
  print_log('save-dir : {:}'.format(save_dir), log)
  print_log('genotype : {:}'.format(genotype), log)
  print_log('config   : {:}'.format(config), log)

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

  checkpoint_path = os.path.join(save_dir, 'checkpoint-{:}.pth'.format(config.data_name))

  Soptimizer = torch.optim.SGD (model.parameters(), lr=config.LR, weight_decay=config.wdecay)
  Aoptimizer = torch.optim.ASGD(model.parameters(), lr=config.LR, t0=0, lambd=0., weight_decay=config.wdecay)
  if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict( checkpoint['state_dict'] )
    Soptimizer.load_state_dict( checkpoint['SGD_optimizer'] )
    Aoptimizer.load_state_dict( checkpoint['ASGD_optimizer'] )
    epoch          = checkpoint['epoch']
    use_asgd       = checkpoint['use_asgd']
    print_log('load checkpoint from {:} and start train from {:}'.format(checkpoint_path, epoch), log)
  else:
    epoch, use_asgd = 0, False

  start_time, epoch_time = time.time(), AverageMeter()
  valid_loss_from_sgd, losses = [], {-1 : 1e9}
  while epoch < config.epochs:
    need_time = convert_secs2time(epoch_time.val * (config.epochs-epoch), True)
    print_log("\n==>>{:s} [Epoch={:04d}/{:04d}] {:}".format(time_string(), epoch, config.epochs, need_time), log)
    if use_asgd : optimizer = Aoptimizer
    else        : optimizer = Soptimizer

    try:
      Dtime, Btime = train(model, optimizer, corpus, train_data, config, epoch, print_freq, log)
    except:
      torch.cuda.empty_cache()
      checkpoint = torch.load(checkpoint_path)
      model.load_state_dict( checkpoint['state_dict'] )
      Soptimizer.load_state_dict( checkpoint['SGD_optimizer'] )
      Aoptimizer.load_state_dict( checkpoint['ASGD_optimizer'] )
      epoch          = checkpoint['epoch']
      use_asgd       = checkpoint['use_asgd']
      valid_loss_from_sgd = checkpoint['valid_loss_from_sgd']
      continue
    if use_asgd:
      tmp = {}
      for prm in model.parameters():
        tmp[prm] = prm.data.clone()
        prm.data = Aoptimizer.state[prm]['ax'].clone()

      val_loss = evaluate(model, corpus, valid_data, config.eval_batch, config.bptt)
    
      for prm in model.parameters():
        prm.data = tmp[prm].clone()
    else:
      val_loss = evaluate(model, corpus, valid_data, config.eval_batch, config.bptt)
      if len(valid_loss_from_sgd) > config.nonmono and val_loss > min(valid_loss_from_sgd):
        use_asgd = True
      valid_loss_from_sgd.append( val_loss )

    print_log('{:} end of epoch {:3d} with {:} | valid loss {:5.2f} | valid ppl {:8.2f}'.format(time_string(), epoch, 'ASGD' if use_asgd else 'SGD', val_loss, math.exp(val_loss)), log)

    if val_loss < min(losses.values()):
      if use_asgd:
        tmp = {}
        for prm in model.parameters():
          tmp[prm] = prm.data.clone()
          prm.data = Aoptimizer.state[prm]['ax'].clone()
      torch.save({'epoch'     : epoch,
                  'use_asgd'  : use_asgd,
                  'valid_loss_from_sgd': valid_loss_from_sgd,
                  'state_dict': model.state_dict(),
                  'SGD_optimizer' : Soptimizer.state_dict(),
                  'ASGD_optimizer': Aoptimizer.state_dict()},
                  checkpoint_path)
      if use_asgd:
        for prm in model.parameters():
          prm.data = tmp[prm].clone()
      print_log('save into {:}'.format(checkpoint_path), log)
      if use_asgd:
        tmp = {}
        for prm in model.parameters():
          tmp[prm] = prm.data.clone()
          prm.data = Aoptimizer.state[prm]['ax'].clone()
      test_loss = evaluate(model, corpus, test_data, config.test_batch, config.bptt)
      if use_asgd:
        for prm in model.parameters():
          prm.data = tmp[prm].clone()
      print_log('| epoch={:03d} | test loss {:5.2f} | test ppl {:8.2f}'.format(epoch, test_loss, math.exp(test_loss)), log)
    losses[epoch] = val_loss
    epoch = epoch + 1
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

      
  print_log('--------------------- Finish Training ----------------', log)
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict( checkpoint['state_dict'] )
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



def train(model, optimizer, corpus, train_data, config, epoch, print_freq, log):
  # Turn on training mode which enables dropout.
  total_loss, data_time, batch_time = 0, AverageMeter(), AverageMeter()
  start_time = time.time()
  ntokens = len(corpus.dictionary)

  hidden_train = model.init_hidden(config.train_batch)
  
  batch, i = 0, 0
  while i < train_data.size(0) - 1 - 1:
    bptt = config.bptt if np.random.random() < 0.95 else config.bptt / 2.
    # Prevent excessively small or negative sequence lengths
    seq_len = max(5, int(np.random.normal(bptt, 5)))
    # There's a very small chance that it could select a very long sequence length resulting in OOM
    seq_len = min(seq_len, config.bptt + config.max_seq_len_delta)
    

    lr2 = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr2 * seq_len / config.bptt
  
    model.train()
    data, targets = get_batch(train_data, i, seq_len)
    targets = targets.contiguous().view(-1)
    # count data preparation time
    data_time.update(time.time() - start_time)

    optimizer.zero_grad()
    hidden_train = repackage_hidden(hidden_train)
    log_prob, hidden_train, rnn_hs, dropped_rnn_hs = model(data, hidden_train, return_h=True)
    raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)

    loss = raw_loss
    # Activiation Regularization
    if config.alpha > 0:
      loss = loss + sum(config.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
    # Temporal Activation Regularization (slowness)
    loss = loss + sum(config.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
    optimizer.step()

    gc.collect()

    optimizer.param_groups[0]['lr'] = lr2

    total_loss += raw_loss.item()
    assert torch.isnan(loss) == False, '--- Epoch={:04d} :: {:03d}/{:03d} Get Loss = Nan'.format(epoch, batch, len(train_data)//config.bptt)

    batch_time.update(time.time() - start_time)
    start_time = time.time()
    batch, i = batch + 1, i + seq_len

    if batch % print_freq == 0:
      cur_loss = total_loss / print_freq
      print_log('  >> Epoch: {:04d} :: {:03d}/{:03d} || loss = {:5.2f}, ppl = {:8.2f}'.format(epoch, batch, len(train_data) // config.bptt, cur_loss, math.exp(cur_loss)), log)
      total_loss = 0
  return data_time.sum, batch_time.sum
