##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
########################################################
# python exps/NAS-Bench-201/test-correlation.py --api_path $HOME/.torch/NAS-Bench-201-v1_0-e61699.pth
########################################################
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces, CellStructure
from nas_201_api  import NASBench201API as API

  
def valid_func(xloader, network, criterion):
  data_time, batch_time = AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.eval()
  end = time.time()
  with torch.no_grad():
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      arch_targets = arch_targets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - end)
      # prediction
      _, logits = network(arch_inputs)
      arch_loss = criterion(logits, arch_targets)
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
  return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  if xargs.dataset == 'cifar10' or xargs.dataset == 'cifar100':
    split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
    cifar_split = load_config(split_Fpath, None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid
    logger.log('Load split file from {:}'.format(split_Fpath))
  elif xargs.dataset.startswith('ImageNet16'):
    split_Fpath = 'configs/nas-benchmark/{:}-split.txt'.format(xargs.dataset)
    imagenet16_split = load_config(split_Fpath, None, None)
    train_split, valid_split = imagenet16_split.train, imagenet16_split.valid
    logger.log('Load split file from {:}'.format(split_Fpath))
  else:
    raise ValueError('invalid dataset : {:}'.format(xargs.dataset))
  config_path = 'configs/nas-benchmark/algos/DARTS.config'
  config = load_config(config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  # To split data
  train_data_v2 = deepcopy(train_data)
  train_data_v2.transform = valid_data.transform
  valid_data    = train_data_v2
  search_data   = SearchDataset(xargs.dataset, train_data, train_split, valid_split)
  # data loader
  search_loader = torch.utils.data.DataLoader(search_data, batch_size=config.batch_size, shuffle=True , num_workers=xargs.workers, pin_memory=True)
  valid_loader  = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=xargs.workers, pin_memory=True)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces('cell', xargs.search_space_name)
  model_config = dict2config({'name': 'DARTS-V2', 'C': xargs.channel, 'N': xargs.num_cells,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space'    : search_space}, None)
  search_model = get_cell_based_tiny_net(model_config)
  logger.log('search-model :\n{:}'.format(search_model))
  
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  #logger.log('{:}'.format(search_model))
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

  logger.close()
  

def check_unique_arch(meta_file):
  api = API(str(meta_file))
  arch_strs = deepcopy(api.meta_archs)
  xarchs = [CellStructure.str2structure(x) for x in arch_strs]
  def get_unique_matrix(archs, consider_zero):
    UniquStrs = [arch.to_unique_str(consider_zero) for arch in archs]
    print ('{:} create unique-string ({:}/{:}) done'.format(time_string(), len(set(UniquStrs)), len(UniquStrs)))
    Unique2Index = dict()
    for index, xstr in enumerate(UniquStrs):
      if xstr not in Unique2Index: Unique2Index[xstr] = list()
      Unique2Index[xstr].append( index )
    sm_matrix = torch.eye(len(archs)).bool()
    for _, xlist in Unique2Index.items():
      for i in xlist:
        for j in xlist:
          sm_matrix[i,j] = True
    unique_ids, unique_num = [-1 for _ in archs], 0
    for i in range(len(unique_ids)):
      if unique_ids[i] > -1: continue
      neighbours = sm_matrix[i].nonzero().view(-1).tolist()
      for nghb in neighbours:
        assert unique_ids[nghb] == -1, 'impossible'
        unique_ids[nghb] = unique_num
      unique_num += 1
    return sm_matrix, unique_ids, unique_num

  print ('There are {:} valid-archs'.format( sum(arch.check_valid() for arch in xarchs) ))
  sm_matrix, uniqueIDs, unique_num = get_unique_matrix(xarchs, None)
  print ('{:} There are {:} unique architectures (considering nothing).'.format(time_string(), unique_num))
  sm_matrix, uniqueIDs, unique_num = get_unique_matrix(xarchs, False)
  print ('{:} There are {:} unique architectures (not considering zero).'.format(time_string(), unique_num))
  sm_matrix, uniqueIDs, unique_num = get_unique_matrix(xarchs,  True)
  print ('{:} There are {:} unique architectures (considering zero).'.format(time_string(), unique_num))


def check_cor_for_bandit(meta_file, test_epoch, use_less_or_not, is_rand=True, need_print=False):
  if isinstance(meta_file, API):
    api = meta_file
  else:
    api = API(str(meta_file))
  cifar10_currs     = []
  cifar10_valid     = []
  cifar10_test      = []
  cifar100_valid    = []
  cifar100_test     = []
  imagenet_test     = []
  imagenet_valid    = []
  for idx, arch in enumerate(api):
    results = api.get_more_info(idx, 'cifar10-valid' , test_epoch-1, use_less_or_not, is_rand)
    cifar10_currs.append( results['valid-accuracy'] )
    # --->>>>>
    results = api.get_more_info(idx, 'cifar10-valid' , None, False, is_rand)
    cifar10_valid.append( results['valid-accuracy'] )
    results = api.get_more_info(idx, 'cifar10'       , None, False, is_rand)
    cifar10_test.append( results['test-accuracy'] )
    results = api.get_more_info(idx, 'cifar100'      , None, False, is_rand)
    cifar100_test.append( results['test-accuracy'] )
    cifar100_valid.append( results['valid-accuracy'] )
    results = api.get_more_info(idx, 'ImageNet16-120', None, False, is_rand)
    imagenet_test.append( results['test-accuracy'] )
    imagenet_valid.append( results['valid-accuracy'] )
  def get_cor(A, B):
    return float(np.corrcoef(A, B)[0,1])
  cors = []
  for basestr, xlist in zip(['C-010-V', 'C-010-T', 'C-100-V', 'C-100-T', 'I16-V', 'I16-T'], [cifar10_valid, cifar10_test, cifar100_valid, cifar100_test, imagenet_valid, imagenet_test]):
    correlation = get_cor(cifar10_currs, xlist)
    if need_print: print ('With {:3d}/{:}-epochs-training, the correlation between cifar10-valid and {:} is : {:}'.format(test_epoch, '012' if use_less_or_not else '200', basestr, correlation))
    cors.append( correlation )
    #print ('With {:3d}/200-epochs-training, the correlation between cifar10-valid and {:} is : {:}'.format(test_epoch, basestr, get_cor(cifar10_valid_200, xlist)))
    #print('-'*200)
  #print('*'*230)
  return cors


def check_cor_for_bandit_v2(meta_file, test_epoch, use_less_or_not, is_rand):
  corrs = []
  for i in tqdm(range(100)):
    x = check_cor_for_bandit(meta_file, test_epoch, use_less_or_not, is_rand, False)
    corrs.append( x )
  #xstrs = ['CIFAR-010', 'C-100-V', 'C-100-T', 'I16-V', 'I16-T']
  xstrs = ['C-010-V', 'C-010-T', 'C-100-V', 'C-100-T', 'I16-V', 'I16-T']
  correlations = np.array(corrs)
  print('------>>>>>>>> {:03d}/{:} >>>>>>>> ------'.format(test_epoch, '012' if use_less_or_not else '200'))
  for idx, xstr in enumerate(xstrs):
    print ('{:8s} ::: mean={:.4f}, std={:.4f} :: {:.4f}\\pm{:.4f}'.format(xstr, correlations[:,idx].mean(), correlations[:,idx].std(), correlations[:,idx].mean(), correlations[:,idx].std()))
  print('')


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Analysis of NAS-Bench-201")
  parser.add_argument('--save_dir',  type=str, default='./output/search-cell-nas-bench-201/visuals', help='The base-name of folder to save checkpoints and log.')
  parser.add_argument('--api_path',  type=str, default=None,                                         help='The path to the NAS-Bench-201 benchmark file.')
  args = parser.parse_args()

  vis_save_dir = Path(args.save_dir)
  vis_save_dir.mkdir(parents=True, exist_ok=True)
  meta_file = Path(args.api_path)
  assert meta_file.exists(), 'invalid path for api : {:}'.format(meta_file)

  #check_unique_arch(meta_file)
  api = API(str(meta_file))
  #for iepoch in [11, 25, 50, 100, 150, 175, 200]:
  #  check_cor_for_bandit(api,  6, iepoch)
  #  check_cor_for_bandit(api, 12, iepoch)
  check_cor_for_bandit_v2(api,   6,  True, True)
  check_cor_for_bandit_v2(api,  12,  True, True)
  check_cor_for_bandit_v2(api,  12, False, True)
  check_cor_for_bandit_v2(api,  24, False, True)
  check_cor_for_bandit_v2(api, 100, False, True)
  check_cor_for_bandit_v2(api, 150, False, True)
  check_cor_for_bandit_v2(api, 175, False, True)
  check_cor_for_bandit_v2(api, 200, False, True)
  print('----')
