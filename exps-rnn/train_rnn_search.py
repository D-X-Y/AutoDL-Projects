import os, gc, sys, math, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from utils import AverageMeter, time_string, convert_secs2time
from utils import print_log, obtain_accuracy
from utils import count_parameters_in_MB
from datasets import Corpus
from nas_rnn import batchify, get_batch, repackage_hidden
from nas_rnn import DARTSCellSearch, RNNModelSearch
from train_rnn_utils import main_procedure
from scheduler import load_config

parser = argparse.ArgumentParser("RNN")
parser.add_argument('--data_path',         type=str,   help='Path to dataset')
parser.add_argument('--emsize',            type=int,   default=300,  help='size of word embeddings')
parser.add_argument('--nhid',              type=int,   default=300,  help='number of hidden units per layer')
parser.add_argument('--nhidlast',          type=int,   default=300,  help='number of hidden units for the last rnn layer')
parser.add_argument('--clip',              type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs',            type=int,   default=50,   help='num of training epochs')
parser.add_argument('--batch_size',        type=int,   default=256,  help='the batch size')
parser.add_argument('--eval_batch_size',   type=int,   default=10,   help='the evaluation batch size')
parser.add_argument('--bptt',              type=int,   default=35,   help='the sequence length')
# DropOut
parser.add_argument('--dropout',           type=float, default=0.75, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth',          type=float, default=0.25, help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx',          type=float, default=0.75, help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti',          type=float, default=0.2,  help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute',          type=float, default=0,    help='dropout to remove words from embedding layer (0 = no dropout)')
# Regularization
parser.add_argument('--lr',                type=float, default=20,   help='initial learning rate')
parser.add_argument('--alpha',             type=float, default=0,    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta',              type=float, default=1e-3, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay',            type=float, default=5e-7, help='weight decay applied to all weights')
# architecture leraning rate
parser.add_argument('--arch_lr',           type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_wdecay',       type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--config_path',       type=str,                 help='the training configure for the discovered model')
# log
parser.add_argument('--save_path',         type=str, help='Folder to save checkpoints and log.')
parser.add_argument('--print_freq',        type=int, help='print frequency (default: 200)')
parser.add_argument('--manualSeed',        type=int, help='manual seed')
args = parser.parse_args()

assert torch.cuda.is_available(), 'torch.cuda is not available'

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
if args.nhidlast < 0:
  args.nhidlast = args.emsize
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

  # Dataset
  corpus = Corpus(args.data_path)
  train_data  = batchify(corpus.train, args.batch_size, True)
  search_data = batchify(corpus.valid, args.batch_size, True)
  valid_data  = batchify(corpus.valid, args.eval_batch_size, True)
  print_log("Train--Data Size : {:}".format(train_data.size()), log)
  print_log("Search-Data Size : {:}".format(search_data.size()), log)
  print_log("Valid--Data Size : {:}".format(valid_data.size()), log)

  ntokens = len(corpus.dictionary)
  model = RNNModelSearch(ntokens, args.emsize, args.nhid, args.nhidlast, 
                          args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute,
                          DARTSCellSearch, None)
  model = model.cuda()
  print_log('model ==>> : {:}'.format(model), log)
  print_log('Parameter size : {:} MB'.format(count_parameters_in_MB(model)), log)
  
  base_optimizer = torch.optim.SGD(model.base_parameters(), lr=args.lr, weight_decay=args.wdecay)
  arch_optimizer = torch.optim.Adam(model.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_wdecay)

  config = load_config(args.config_path)
  print_log('Load config from {:} ==>>\n  {:}'.format(args.config_path, config), log)

  # snapshot
  checkpoint_path = os.path.join(args.save_path, 'checkpoint-search.pth')
  if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict( checkpoint['state_dict'] )
    base_optimizer.load_state_dict( checkpoint['base_optimizer'] )
    arch_optimizer.load_state_dict( checkpoint['arch_optimizer'] )
    genotypes    = checkpoint['genotypes']
    valid_losses = checkpoint['valid_losses']
    print_log('Load checkpoint from {:} with start-epoch = {:}'.format(checkpoint_path, start_epoch), log)
  else:
    start_epoch, genotypes, valid_losses = 0, {}, {-1:1e8}
    print_log('Train model-search from scratch.', log)

  # Main loop
  start_time, epoch_time, total_train_time = time.time(), AverageMeter(), 0
  for epoch in range(start_epoch, args.epochs):

    need_time = convert_secs2time(epoch_time.val * (args.epochs-epoch), True)
    print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs, need_time), log)
    # training
    data_time, train_time = train(model, base_optimizer, arch_optimizer, corpus, train_data, search_data, epoch, log)
    total_train_time += train_time
    # evaluation

    # validation
    valid_loss = infer(model, corpus, valid_data, args.eval_batch_size)
    # save genotype
    if valid_loss < min( valid_losses.values() ): is_best = True
    else                                        : is_best = False
    print_log('-'*10 + ' [Epoch={:03d}/{:03d}] : is-best={:}, validation-loss={:}, validation-PPL={:}'.format(epoch, args.epochs, is_best, valid_loss, math.exp(valid_loss)), log)

    valid_losses[epoch] = valid_loss
    genotypes[epoch] = model.genotype()
    print_log(' the {:}-th genotype = {:}'.format(epoch, genotypes[epoch]), log)
    # save checkpoint
    if is_best:
      genotypes['best'] = model.genotype()
      torch.save({'epoch' : epoch + 1,
                'args'  : deepcopy(args),
                'state_dict': model.state_dict(),
                'genotypes' : genotypes,
                'valid_losses'   : valid_losses,
                'base_optimizer' : base_optimizer.state_dict(),
                'arch_optimizer' : arch_optimizer.state_dict()},
                checkpoint_path)
    print_log('----> Save into {:}'.format(checkpoint_path), log)


    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  print_log('Finish with training time = {:}'.format( convert_secs2time(total_train_time, True) ), log)

  # clear GPU cache
  torch.cuda.empty_cache()
  main_procedure(config, genotypes['best'], args.save_path, args.print_freq, log)
  log.close()


def train(model, base_optimizer, arch_optimizer, corpus, train_data, search_data, epoch, log):

  data_time, batch_time = AverageMeter(), AverageMeter()
  # Turn on training mode which enables dropout.
  total_loss = 0
  start_time = time.time()
  ntokens = len(corpus.dictionary)
  hidden_train, hidden_valid = model.init_hidden(args.batch_size), model.init_hidden(args.batch_size)

  batch, i = 0, 0
  
  while i < train_data.size(0) - 1 - 1:
    seq_len = int( args.bptt if np.random.random() < 0.95 else args.bptt / 2. )
    # Prevent excessively small or negative sequence lengths
    # seq_len = max(5, int(np.random.normal(bptt, 5)))
    # # There's a very small chance that it could select a very long sequence length resulting in OOM
    # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
    for param_group in base_optimizer.param_groups:
      param_group['lr'] *= float( seq_len / args.bptt )

    model.train()

    data_valid, targets_valid = get_batch(search_data, i % (search_data.size(0) - 1), args.bptt)
    data_train, targets_train = get_batch(train_data , i, seq_len)

    hidden_train = repackage_hidden(hidden_train)
    hidden_valid = repackage_hidden(hidden_valid)

    data_time.update(time.time() - start_time)

    # validation loss
    targets_valid = targets_valid.contiguous().view(-1)
  
    arch_optimizer.step()
    log_prob, hidden_valid = model(data_valid, hidden_valid, return_h=False)
    arch_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets_valid)
    arch_loss.backward()
    arch_optimizer.step()

    # model update
    base_optimizer.zero_grad()
    targets_train = targets_train.contiguous().view(-1)

    log_prob, hidden_train, rnn_hs, dropped_rnn_hs = model(data_train, hidden_train, return_h=True)
    raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets_train)

    loss = raw_loss
    # Activiation Regularization
    if args.alpha > 0:
      loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
    # Temporal Activation Regularization (slowness)
    loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
    nn.utils.clip_grad_norm_(model.base_parameters(), args.clip)
    base_optimizer.step()

    for param_group in base_optimizer.param_groups:
      param_group['lr'] /= float( seq_len / args.bptt )

    total_loss += raw_loss.item()
    gc.collect()
  
    batch_time.update(time.time() - start_time)
    start_time = time.time()
    batch, i = batch + 1, i + seq_len

    if batch % args.print_freq == 0 or i >= train_data.size(0) - 1 - 1:
      print_log('  || Epoch: {:03d} :: {:03d}/{:03d}'.format(epoch, batch, len(train_data) // args.bptt), log)
      #print_log('  || Epoch: {:03d} :: {:03d}/{:03d} = {:}'.format(epoch, batch, len(train_data) // args.bptt, model.genotype()), log)
      cur_loss = total_loss / args.print_freq
      print_log('     ---> Time : data {:.3f} ({:.3f}) batch {:.3f} ({:.3f}) Loss : {:}, PPL : {:}'.format(data_time.val, data_time.avg, batch_time.val, batch_time.avg, cur_loss, math.exp(cur_loss)), log)
      print(F.softmax(model.arch_weights, dim=-1))
      total_loss = 0

  return data_time.sum, batch_time.sum


def infer(model, corpus, data_source, batch_size):
  
  model.eval()
  with torch.no_grad():
    total_loss, total_length = 0, 0
    ntokens = len(corpus.dictionary)
    hidden  = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
      data, targets = get_batch(data_source, i, args.bptt)
      targets = targets.view(-1)

      log_prob, hidden = model(data, hidden)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)

      total_loss += loss.item() * len(data)
      total_length += len(data)
      hidden = repackage_hidden(hidden)
  return total_loss / total_length


if __name__ == '__main__':
  main() 
