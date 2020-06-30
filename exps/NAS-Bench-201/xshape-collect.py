#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
# python exps/NAS-Bench-201/xshape-collect.py
#####################################################
import os, re, sys, time, argparse, collections
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, Any, Text, List
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from log_utils    import AverageMeter, time_string, convert_secs2time
from config_utils import dict2config
# NAS-Bench-201 related module or function
from models       import CellStructure, get_cell_based_tiny_net
from nas_201_api  import NASBench301API, ArchResults, ResultsCount
from procedures   import bench_pure_evaluate as pure_evaluate, get_nas_bench_loaders


def account_one_arch(arch_index: int, arch_str: Text, checkpoints: List[Text], datasets: List[Text]) -> ArchResults:
  information = ArchResults(arch_index, arch_str)

  for checkpoint_path in checkpoints:
    try:
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except:
      raise ValueError('This checkpoint failed to be loaded : {:}'.format(checkpoint_path))
    used_seed  = checkpoint_path.name.split('-')[-1].split('.')[0]
    ok_dataset = 0
    for dataset in datasets:
      if dataset not in checkpoint:
        print('Can not find {:} in arch-{:} from {:}'.format(dataset, arch_index, checkpoint_path))
        continue
      else:
        ok_dataset += 1
      results     = checkpoint[dataset]
      assert results['finish-train'], 'This {:} arch seed={:} does not finish train on {:} ::: {:}'.format(arch_index, used_seed, dataset, checkpoint_path)
      arch_config = {'name': 'infer.shape.tiny', 'channels': arch_str, 'arch_str': arch_str,
                     'genotype': results['arch_config']['genotype'],
                     'class_num': results['arch_config']['num_classes']}
      xresult = ResultsCount(dataset, results['net_state_dict'], results['train_acc1es'], results['train_losses'],
                             results['param'], results['flop'], arch_config, used_seed, results['total_epoch'], None)
      xresult.update_train_info(results['train_acc1es'], results['train_acc5es'], results['train_losses'], results['train_times'])
      xresult.update_eval(results['valid_acc1es'], results['valid_losses'], results['valid_times'])
      information.update(dataset, int(used_seed), xresult)
    if ok_dataset < len(datasets): raise ValueError('{:} does find enought data : {:} vs {:}'.format(checkpoint_path, ok_dataset, len(datasets)))
  return information


def correct_time_related_info(hp2info: Dict[Text, ArchResults]):
  # calibrate the latency based on the number of epochs = 01, since they are trained on the same machine.
  x1 = hp2info['01'].get_metrics('cifar10-valid', 'x-valid')['all_time'] / 98
  x2 = hp2info['01'].get_metrics('cifar10-valid', 'ori-test')['all_time'] / 40
  cifar010_latency = (x1 + x2) / 2
  for hp, arch_info in hp2info.items():
    arch_info.reset_latency('cifar10-valid', None, cifar010_latency)
    arch_info.reset_latency('cifar10', None, cifar010_latency)
  # hp2info['01'].get_latency('cifar10')

  x1 = hp2info['01'].get_metrics('cifar100', 'ori-test')['all_time'] / 40
  x2 = hp2info['01'].get_metrics('cifar100', 'x-test')['all_time'] / 20
  x3 = hp2info['01'].get_metrics('cifar100', 'x-valid')['all_time'] / 20
  cifar100_latency = (x1 + x2 + x3) / 3
  for hp, arch_info in hp2info.items():
    arch_info.reset_latency('cifar100', None, cifar100_latency)

  x1 = hp2info['01'].get_metrics('ImageNet16-120', 'ori-test')['all_time'] / 24
  x2 = hp2info['01'].get_metrics('ImageNet16-120', 'x-test')['all_time'] / 12
  x3 = hp2info['01'].get_metrics('ImageNet16-120', 'x-valid')['all_time'] / 12
  image_latency = (x1 + x2 + x3) / 3
  for hp, arch_info in hp2info.items():
    arch_info.reset_latency('ImageNet16-120', None, image_latency)

  # CIFAR10 VALID
  train_per_epoch_time = list(hp2info['01'].query('cifar10-valid', 777).train_times.values())
  train_per_epoch_time = sum(train_per_epoch_time) / len(train_per_epoch_time)
  eval_ori_test_time, eval_x_valid_time = [], []
  for key, value in hp2info['01'].query('cifar10-valid', 777).eval_times.items():
    if key.startswith('ori-test@'):
      eval_ori_test_time.append(value)
    elif key.startswith('x-valid@'):
      eval_x_valid_time.append(value)
    else: raise ValueError('-- {:} --'.format(key))
  eval_ori_test_time = sum(eval_ori_test_time) / len(eval_ori_test_time)
  eval_x_valid_time = sum(eval_x_valid_time) / len(eval_x_valid_time)
  for hp, arch_info in hp2info.items():
    arch_info.reset_pseudo_train_times('cifar10-valid', None, train_per_epoch_time)
    arch_info.reset_pseudo_eval_times('cifar10-valid', None, 'x-valid', eval_x_valid_time)
    arch_info.reset_pseudo_eval_times('cifar10-valid', None, 'ori-test', eval_ori_test_time)

  # CIFAR10
  train_per_epoch_time = list(hp2info['01'].query('cifar10', 777).train_times.values())
  train_per_epoch_time = sum(train_per_epoch_time) / len(train_per_epoch_time)
  eval_ori_test_time = []
  for key, value in hp2info['01'].query('cifar10', 777).eval_times.items():
    if key.startswith('ori-test@'):
      eval_ori_test_time.append(value)
    else: raise ValueError('-- {:} --'.format(key))
  eval_ori_test_time = sum(eval_ori_test_time) / len(eval_ori_test_time)
  for hp, arch_info in hp2info.items():
    arch_info.reset_pseudo_train_times('cifar10', None, train_per_epoch_time)
    arch_info.reset_pseudo_eval_times('cifar10', None, 'ori-test', eval_ori_test_time)

  # CIFAR100
  train_per_epoch_time = list(hp2info['01'].query('cifar100', 777).train_times.values())
  train_per_epoch_time = sum(train_per_epoch_time) / len(train_per_epoch_time)
  eval_ori_test_time, eval_x_valid_time, eval_x_test_time = [], [], []
  for key, value in hp2info['01'].query('cifar100', 777).eval_times.items():
    if key.startswith('ori-test@'):
      eval_ori_test_time.append(value)
    elif key.startswith('x-valid@'):
      eval_x_valid_time.append(value)
    elif key.startswith('x-test@'):
      eval_x_test_time.append(value)
    else: raise ValueError('-- {:} --'.format(key))
  eval_ori_test_time = sum(eval_ori_test_time) / len(eval_ori_test_time)
  eval_x_valid_time = sum(eval_x_valid_time) / len(eval_x_valid_time)
  eval_x_test_time = sum(eval_x_test_time) / len(eval_x_test_time)
  for hp, arch_info in hp2info.items():
    arch_info.reset_pseudo_train_times('cifar100', None, train_per_epoch_time)
    arch_info.reset_pseudo_eval_times('cifar100', None, 'x-valid', eval_x_valid_time)
    arch_info.reset_pseudo_eval_times('cifar100', None, 'x-test', eval_x_test_time)
    arch_info.reset_pseudo_eval_times('cifar100', None, 'ori-test', eval_ori_test_time)

  # ImageNet16-120
  train_per_epoch_time = list(hp2info['01'].query('ImageNet16-120', 777).train_times.values())
  train_per_epoch_time = sum(train_per_epoch_time) / len(train_per_epoch_time)
  eval_ori_test_time, eval_x_valid_time, eval_x_test_time = [], [], []
  for key, value in hp2info['01'].query('ImageNet16-120', 777).eval_times.items():
    if key.startswith('ori-test@'):
      eval_ori_test_time.append(value)
    elif key.startswith('x-valid@'):
      eval_x_valid_time.append(value)
    elif key.startswith('x-test@'):
      eval_x_test_time.append(value)
    else: raise ValueError('-- {:} --'.format(key))
  eval_ori_test_time = sum(eval_ori_test_time) / len(eval_ori_test_time)
  eval_x_valid_time = sum(eval_x_valid_time) / len(eval_x_valid_time)
  eval_x_test_time = sum(eval_x_test_time) / len(eval_x_test_time)
  for hp, arch_info in hp2info.items():
    arch_info.reset_pseudo_train_times('ImageNet16-120', None, train_per_epoch_time)
    arch_info.reset_pseudo_eval_times('ImageNet16-120', None, 'x-valid', eval_x_valid_time)
    arch_info.reset_pseudo_eval_times('ImageNet16-120', None, 'x-test', eval_x_test_time)
    arch_info.reset_pseudo_eval_times('ImageNet16-120', None, 'ori-test', eval_ori_test_time)
  return hp2info


def simplify(save_dir, save_name, nets, total):
  
  hps, seeds = ['01', '12', '90'], set()
  for hp in hps:
    sub_save_dir = save_dir / 'raw-data-{:}'.format(hp)
    ckps = sorted(list(sub_save_dir.glob('arch-*-seed-*.pth')))
    seed2names = defaultdict(list)
    for ckp in ckps:
      parts = re.split('-|\.', ckp.name)
      seed2names[parts[3]].append(ckp.name)
    print('DIR : {:}'.format(sub_save_dir))
    nums = []
    for seed, xlist in seed2names.items():
      seeds.add(seed)
      nums.append(len(xlist))
      print('  seed={:}, num={:}'.format(seed, len(xlist)))
    # assert len(nets) == total == max(nums), 'there are some missed files : {:} vs {:}'.format(max(nums), total)
  print('{:} start simplify the checkpoint.'.format(time_string()))

  datasets = ('cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120')

  simplify_save_dir, arch2infos, evaluated_indexes = save_dir / save_name, {}, set()
  simplify_save_dir.mkdir(parents=True, exist_ok=True)
  end_time, arch_time = time.time(), AverageMeter()
  # for index, arch_str in enumerate(nets):
  for index in tqdm(range(total)):
    arch_str = nets[index]
    hp2info = OrderedDict()
    for hp in hps:
      sub_save_dir = save_dir / 'raw-data-{:}'.format(hp)
      ckps = [sub_save_dir / 'arch-{:06d}-seed-{:}.pth'.format(index, seed) for seed in seeds]
      ckps = [x for x in ckps if x.exists()]
      if len(ckps) == 0: raise ValueError('Invalid data : index={:}, hp={:}'.format(index, hp))
      
      arch_info = account_one_arch(index, arch_str, ckps, datasets)
      hp2info[hp] = arch_info
    
    hp2info = correct_time_related_info(hp2info)
    evaluated_indexes.add(index)

    to_save_data = OrderedDict({'01': hp2info['01'].state_dict(),
                                '12': hp2info['12'].state_dict(),
                                '90': hp2info['90'].state_dict()})
    torch.save(to_save_data, simplify_save_dir / '{:}-FULL.pth'.format(index))
    
    for hp in hps: hp2info[hp].clear_params()
    to_save_data = OrderedDict({'01': hp2info['01'].state_dict(),
                                '12': hp2info['12'].state_dict(),
                                '90': hp2info['90'].state_dict()})
    torch.save(to_save_data, simplify_save_dir / '{:}-SIMPLE.pth'.format(index))
    arch2infos[index] = to_save_data
    # measure elapsed time
    arch_time.update(time.time() - end_time)
    end_time  = time.time()
    need_time = '{:}'.format(convert_secs2time(arch_time.avg * (total-index-1), True))
    # print('{:} {:06d}/{:06d} : still need {:}'.format(time_string(), index, total, need_time))
  print('{:} {:} done.'.format(time_string(), save_name))
  final_infos = {'meta_archs' : nets,
                 'total_archs': total,
                 'arch2infos' : arch2infos,
                 'evaluated_indexes': evaluated_indexes}
  save_file_name = save_dir / '{:}.pth'.format(save_name)
  torch.save(final_infos, save_file_name)
  print ('Save {:} / {:} architecture results into {:}.'.format(len(evaluated_indexes), total, save_file_name))


def traverse_net(candidates: List[int], N: int):
  nets = ['']
  for i in range(N):
    new_nets = []
    for net in nets:
      for C in candidates:
        new_nets.append(str(C) if net == '' else "{:}:{:}".format(net,C))
    nets = new_nets
  return nets


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='NAS-BENCH-201', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--base_save_dir',  type=str, default='./output/NAS-BENCH-202',    help='The base-name of folder to save checkpoints and log.')
  parser.add_argument('--candidateC'   ,  type=int, nargs='+', default=[8, 16, 24, 32, 40, 48, 56, 64], help='.')
  parser.add_argument('--num_layers'   ,  type=int, default=5,      help='The number of layers in a network.')
  parser.add_argument('--check_N'      ,  type=int, default=32768,  help='For safety.')
  parser.add_argument('--save_name'    ,  type=str, default='simplify',                  help='The save directory.')
  args = parser.parse_args()
  
  nets = traverse_net(args.candidateC, args.num_layers)
  if len(nets) != args.check_N: raise ValueError('Pre-num-check failed : {:} vs {:}'.format(len(nets), args.check_N))

  save_dir  = Path(args.base_save_dir)
  simplify(save_dir, args.save_name, nets, args.check_N)
