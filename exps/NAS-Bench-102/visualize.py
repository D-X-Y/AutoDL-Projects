##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# python exps/NAS-Bench-102/visualize.py --api_path $HOME/.torch/NAS-Bench-102-v1_0-e61699.pth
##################################################
import os, sys, time, argparse, collections
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
import matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('agg')
import matplotlib.pyplot as plt

lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from log_utils    import time_string
from nas_102_api  import NASBench102API as API



def calculate_correlation(*vectors):
  matrix = []
  for i, vectori in enumerate(vectors):
    x = []
    for j, vectorj in enumerate(vectors):
      x.append( np.corrcoef(vectori, vectorj)[0,1] )
    matrix.append( x )
  return np.array(matrix)



def visualize_relative_ranking(vis_save_dir):
  print ('\n' + '-'*100)
  cifar010_cache_path = vis_save_dir / '{:}-cache-info.pth'.format('cifar10')
  cifar100_cache_path = vis_save_dir / '{:}-cache-info.pth'.format('cifar100')
  imagenet_cache_path = vis_save_dir / '{:}-cache-info.pth'.format('ImageNet16-120')
  cifar010_info = torch.load(cifar010_cache_path)
  cifar100_info = torch.load(cifar100_cache_path)
  imagenet_info = torch.load(imagenet_cache_path)
  indexes       = list(range(len(cifar010_info['params'])))

  print ('{:} start to visualize relative ranking'.format(time_string()))
  # maximum accuracy with ResNet-level params 11472
  x_010_accs    = [ cifar010_info['test_accs'][i] if cifar010_info['params'][i] <= cifar010_info['params'][11472] else -1 for i in indexes]
  x_100_accs    = [ cifar100_info['test_accs'][i] if cifar100_info['params'][i] <= cifar100_info['params'][11472] else -1 for i in indexes]
  x_img_accs    = [ imagenet_info['test_accs'][i] if imagenet_info['params'][i] <= imagenet_info['params'][11472] else -1 for i in indexes]
 
  cifar010_ord_indexes = sorted(indexes, key=lambda i: cifar010_info['test_accs'][i])
  cifar100_ord_indexes = sorted(indexes, key=lambda i: cifar100_info['test_accs'][i])
  imagenet_ord_indexes = sorted(indexes, key=lambda i: imagenet_info['test_accs'][i])

  cifar100_labels, imagenet_labels = [], []
  for idx in cifar010_ord_indexes:
    cifar100_labels.append( cifar100_ord_indexes.index(idx) )
    imagenet_labels.append( imagenet_ord_indexes.index(idx) )
  print ('{:} prepare data done.'.format(time_string()))

  dpi, width, height = 300, 2600, 2600
  figsize = width / float(dpi), height / float(dpi)
  LabelSize, LegendFontsize = 18, 18
  resnet_scale, resnet_alpha = 120, 0.5

  fig = plt.figure(figsize=figsize)
  ax  = fig.add_subplot(111)
  plt.xlim(min(indexes), max(indexes))
  plt.ylim(min(indexes), max(indexes))
  #plt.ylabel('y').set_rotation(0)
  plt.yticks(np.arange(min(indexes), max(indexes), max(indexes)//6), fontsize=LegendFontsize, rotation='vertical')
  plt.xticks(np.arange(min(indexes), max(indexes), max(indexes)//6), fontsize=LegendFontsize)
  #ax.scatter(indexes, cifar100_labels, marker='^', s=0.5, c='tab:green', alpha=0.8, label='CIFAR-100')
  #ax.scatter(indexes, imagenet_labels, marker='*', s=0.5, c='tab:red'  , alpha=0.8, label='ImageNet-16-120')
  #ax.scatter(indexes, indexes        , marker='o', s=0.5, c='tab:blue' , alpha=0.8, label='CIFAR-10')
  ax.scatter(indexes, cifar100_labels, marker='^', s=0.5, c='tab:green', alpha=0.8)
  ax.scatter(indexes, imagenet_labels, marker='*', s=0.5, c='tab:red'  , alpha=0.8)
  ax.scatter(indexes, indexes        , marker='o', s=0.5, c='tab:blue' , alpha=0.8)
  ax.scatter([-1], [-1], marker='o', s=100, c='tab:blue' , label='CIFAR-10')
  ax.scatter([-1], [-1], marker='^', s=100, c='tab:green', label='CIFAR-100')
  ax.scatter([-1], [-1], marker='*', s=100, c='tab:red'  , label='ImageNet-16-120')
  plt.grid(zorder=0)
  ax.set_axisbelow(True)
  plt.legend(loc=0, fontsize=LegendFontsize)
  ax.set_xlabel('architecture ranking in CIFAR-10', fontsize=LabelSize)
  ax.set_ylabel('architecture ranking', fontsize=LabelSize)
  save_path = (vis_save_dir / 'relative-rank.pdf').resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
  save_path = (vis_save_dir / 'relative-rank.png').resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
  print ('{:} save into {:}'.format(time_string(), save_path))

  # calculate correlation
  sns_size = 15
  CoRelMatrix = calculate_correlation(cifar010_info['valid_accs'], cifar010_info['test_accs'], cifar100_info['valid_accs'], cifar100_info['test_accs'], imagenet_info['valid_accs'], imagenet_info['test_accs'])
  fig = plt.figure(figsize=figsize)
  plt.axis('off')
  h = sns.heatmap(CoRelMatrix, annot=True, annot_kws={'size':sns_size}, fmt='.3f', linewidths=0.5)  
  save_path = (vis_save_dir / 'co-relation-all.pdf').resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
  print ('{:} save into {:}'.format(time_string(), save_path))

  # calculate correlation
  acc_bars = [92, 93]
  for acc_bar in acc_bars:
    selected_indexes = []
    for i, acc in enumerate(cifar010_info['test_accs']):
      if acc > acc_bar: selected_indexes.append( i )
    print ('select {:} architectures'.format(len(selected_indexes)))
    cifar010_valid_accs = np.array(cifar010_info['valid_accs'])[ selected_indexes ]
    cifar010_test_accs  = np.array(cifar010_info['test_accs']) [ selected_indexes ]
    cifar100_valid_accs = np.array(cifar100_info['valid_accs'])[ selected_indexes ]
    cifar100_test_accs  = np.array(cifar100_info['test_accs']) [ selected_indexes ]
    imagenet_valid_accs = np.array(imagenet_info['valid_accs'])[ selected_indexes ]
    imagenet_test_accs  = np.array(imagenet_info['test_accs']) [ selected_indexes ]
    CoRelMatrix = calculate_correlation(cifar010_valid_accs, cifar010_test_accs, cifar100_valid_accs, cifar100_test_accs, imagenet_valid_accs, imagenet_test_accs)
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    h = sns.heatmap(CoRelMatrix, annot=True, annot_kws={'size':sns_size}, fmt='.3f', linewidths=0.5)
    save_path = (vis_save_dir / 'co-relation-top-{:}.pdf'.format(len(selected_indexes))).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
    print ('{:} save into {:}'.format(time_string(), save_path))
  plt.close('all')



def visualize_info(meta_file, dataset, vis_save_dir):
  print ('{:} start to visualize {:} information'.format(time_string(), dataset))
  cache_file_path = vis_save_dir / '{:}-cache-info.pth'.format(dataset)
  if not cache_file_path.exists():
    print ('Do not find cache file : {:}'.format(cache_file_path))
    nas_bench = API(str(meta_file))
    params, flops, train_accs, valid_accs, test_accs, otest_accs = [], [], [], [], [], []
    for index in range( len(nas_bench) ):
      info = nas_bench.query_by_index(index, use_12epochs_result=False)
      resx = info.get_comput_costs(dataset) ; flop, param = resx['flops'], resx['params']
      if dataset == 'cifar10':
        res = info.get_metrics('cifar10', 'train')         ; train_acc = res['accuracy']
        res = info.get_metrics('cifar10-valid', 'x-valid') ; valid_acc = res['accuracy']
        res = info.get_metrics('cifar10', 'ori-test')      ; test_acc  = res['accuracy']
        res = info.get_metrics('cifar10', 'ori-test')      ; otest_acc = res['accuracy']
      else:
        res = info.get_metrics(dataset, 'train')    ; train_acc = res['accuracy']
        res = info.get_metrics(dataset, 'x-valid')  ; valid_acc = res['accuracy']
        res = info.get_metrics(dataset, 'x-test')   ; test_acc  = res['accuracy']
        res = info.get_metrics(dataset, 'ori-test') ; otest_acc = res['accuracy']
      if index == 11472: # resnet
        resnet = {'params':param, 'flops': flop, 'index': 11472, 'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc, 'otest_acc': otest_acc}
      flops.append( flop )
      params.append( param )
      train_accs.append( train_acc )
      valid_accs.append( valid_acc )
      test_accs.append( test_acc )
      otest_accs.append( otest_acc )
    #resnet = {'params': 0.559, 'flops': 78.56, 'index': 11472, 'train_acc': 99.99, 'valid_acc': 90.84, 'test_acc': 93.97}
    info = {'params': params, 'flops': flops, 'train_accs': train_accs, 'valid_accs': valid_accs, 'test_accs': test_accs, 'otest_accs': otest_accs}
    info['resnet'] = resnet
    torch.save(info, cache_file_path)
  else:
    print ('Find cache file : {:}'.format(cache_file_path))
    info = torch.load(cache_file_path)
    params, flops, train_accs, valid_accs, test_accs, otest_accs = info['params'], info['flops'], info['train_accs'], info['valid_accs'], info['test_accs'], info['otest_accs']
    resnet = info['resnet']
  print ('{:} collect data done.'.format(time_string()))

  indexes = list(range(len(params)))
  dpi, width, height = 300, 2600, 2600
  figsize = width / float(dpi), height / float(dpi)
  LabelSize, LegendFontsize = 22, 22
  resnet_scale, resnet_alpha = 120, 0.5

  fig = plt.figure(figsize=figsize)
  ax  = fig.add_subplot(111)
  plt.xticks(np.arange(0, 1.6, 0.3), fontsize=LegendFontsize)
  if dataset == 'cifar10':
    plt.ylim(50, 100)
    plt.yticks(np.arange(50, 101, 10), fontsize=LegendFontsize)
  elif dataset == 'cifar100':
    plt.ylim(25,  75)
    plt.yticks(np.arange(25, 76, 10), fontsize=LegendFontsize)
  else:
    plt.ylim(0, 50)
    plt.yticks(np.arange(0, 51, 10), fontsize=LegendFontsize)
  ax.scatter(params, valid_accs, marker='o', s=0.5, c='tab:blue') 
  ax.scatter([resnet['params']], [resnet['valid_acc']], marker='*', s=resnet_scale, c='tab:orange', label='resnet', alpha=0.4) 
  plt.grid(zorder=0)
  ax.set_axisbelow(True)
  plt.legend(loc=4, fontsize=LegendFontsize)
  ax.set_xlabel('#parameters (MB)', fontsize=LabelSize)
  ax.set_ylabel('the validation accuracy (%)', fontsize=LabelSize)
  save_path = (vis_save_dir / '{:}-param-vs-valid.pdf'.format(dataset)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
  save_path = (vis_save_dir / '{:}-param-vs-valid.png'.format(dataset)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
  print ('{:} save into {:}'.format(time_string(), save_path))

  fig = plt.figure(figsize=figsize)
  ax  = fig.add_subplot(111)
  plt.xticks(np.arange(0, 1.6, 0.3), fontsize=LegendFontsize)
  if dataset == 'cifar10':
    plt.ylim(50, 100)
    plt.yticks(np.arange(50, 101, 10), fontsize=LegendFontsize)
  elif dataset == 'cifar100':
    plt.ylim(25,  75)
    plt.yticks(np.arange(25, 76, 10), fontsize=LegendFontsize)
  else:
    plt.ylim(0, 50)
    plt.yticks(np.arange(0, 51, 10), fontsize=LegendFontsize)
  ax.scatter(params,  test_accs, marker='o', s=0.5, c='tab:blue')
  ax.scatter([resnet['params']], [resnet['test_acc']], marker='*', s=resnet_scale, c='tab:orange', label='resnet', alpha=resnet_alpha)
  plt.grid()
  ax.set_axisbelow(True)
  plt.legend(loc=4, fontsize=LegendFontsize)
  ax.set_xlabel('#parameters (MB)', fontsize=LabelSize)
  ax.set_ylabel('the test accuracy (%)', fontsize=LabelSize)
  save_path = (vis_save_dir / '{:}-param-vs-test.pdf'.format(dataset)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
  save_path = (vis_save_dir / '{:}-param-vs-test.png'.format(dataset)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
  print ('{:} save into {:}'.format(time_string(), save_path))

  fig = plt.figure(figsize=figsize)
  ax  = fig.add_subplot(111)
  plt.xticks(np.arange(0, 1.6, 0.3), fontsize=LegendFontsize)
  if dataset == 'cifar10':
    plt.ylim(50, 100)
    plt.yticks(np.arange(50, 101, 10), fontsize=LegendFontsize)
  elif dataset == 'cifar100':
    plt.ylim(20, 100)
    plt.yticks(np.arange(20, 101, 10), fontsize=LegendFontsize)
  else:
    plt.ylim(25,  76)
    plt.yticks(np.arange(25,  76, 10), fontsize=LegendFontsize)
  ax.scatter(params, train_accs, marker='o', s=0.5, c='tab:blue')
  ax.scatter([resnet['params']], [resnet['train_acc']], marker='*', s=resnet_scale, c='tab:orange', label='resnet', alpha=resnet_alpha)
  plt.grid()
  ax.set_axisbelow(True)
  plt.legend(loc=4, fontsize=LegendFontsize)
  ax.set_xlabel('#parameters (MB)', fontsize=LabelSize)
  ax.set_ylabel('the trarining accuracy (%)', fontsize=LabelSize)
  save_path = (vis_save_dir / '{:}-param-vs-train.pdf'.format(dataset)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
  save_path = (vis_save_dir / '{:}-param-vs-train.png'.format(dataset)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
  print ('{:} save into {:}'.format(time_string(), save_path))

  fig = plt.figure(figsize=figsize)
  ax  = fig.add_subplot(111)
  plt.xlim(0, max(indexes))
  plt.xticks(np.arange(min(indexes), max(indexes), max(indexes)//5), fontsize=LegendFontsize)
  if dataset == 'cifar10':
    plt.ylim(50, 100)
    plt.yticks(np.arange(50, 101, 10), fontsize=LegendFontsize)
  elif dataset == 'cifar100':
    plt.ylim(25,  75)
    plt.yticks(np.arange(25, 76, 10), fontsize=LegendFontsize)
  else:
    plt.ylim(0, 50)
    plt.yticks(np.arange(0, 51, 10), fontsize=LegendFontsize)
  ax.scatter(indexes, test_accs, marker='o', s=0.5, c='tab:blue')
  ax.scatter([resnet['index']], [resnet['test_acc']], marker='*', s=resnet_scale, c='tab:orange', label='resnet', alpha=resnet_alpha)
  plt.grid()
  ax.set_axisbelow(True)
  plt.legend(loc=4, fontsize=LegendFontsize)
  ax.set_xlabel('architecture ID', fontsize=LabelSize)
  ax.set_ylabel('the test accuracy (%)', fontsize=LabelSize)
  save_path = (vis_save_dir / '{:}-test-over-ID.pdf'.format(dataset)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
  save_path = (vis_save_dir / '{:}-test-over-ID.png'.format(dataset)).resolve()
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
  print ('{:} save into {:}'.format(time_string(), save_path))
  plt.close('all')



def visualize_rank_over_time(meta_file, vis_save_dir):
  print ('\n' + '-'*150)
  vis_save_dir.mkdir(parents=True, exist_ok=True)
  print ('{:} start to visualize rank-over-time into {:}'.format(time_string(), vis_save_dir))
  cache_file_path = vis_save_dir / 'rank-over-time-cache-info.pth'
  if not cache_file_path.exists():
    print ('Do not find cache file : {:}'.format(cache_file_path))
    nas_bench = API(str(meta_file))
    print ('{:} load nas_bench done'.format(time_string()))
    params, flops, train_accs, valid_accs, test_accs, otest_accs = [], [], defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    #for iepoch in range(200): for index in range( len(nas_bench) ):
    for index in tqdm(range(len(nas_bench))):
      info = nas_bench.query_by_index(index, use_12epochs_result=False)
      for iepoch in range(200):
        res = info.get_metrics('cifar10'      , 'train'   , iepoch) ; train_acc = res['accuracy']
        res = info.get_metrics('cifar10-valid', 'x-valid' , iepoch) ; valid_acc = res['accuracy']
        res = info.get_metrics('cifar10'      , 'ori-test', iepoch) ; test_acc  = res['accuracy']
        res = info.get_metrics('cifar10'      , 'ori-test', iepoch) ; otest_acc = res['accuracy']
        train_accs[iepoch].append( train_acc )
        valid_accs[iepoch].append( valid_acc )
        test_accs [iepoch].append( test_acc )
        otest_accs[iepoch].append( otest_acc )
        if iepoch == 0:
          res = info.get_comput_costs('cifar10') ; flop, param = res['flops'], res['params']
          flops.append( flop )
          params.append( param )
    info = {'params': params, 'flops': flops, 'train_accs': train_accs, 'valid_accs': valid_accs, 'test_accs': test_accs, 'otest_accs': otest_accs}
    torch.save(info, cache_file_path)
  else:
    print ('Find cache file : {:}'.format(cache_file_path))
    info = torch.load(cache_file_path)
    params, flops, train_accs, valid_accs, test_accs, otest_accs = info['params'], info['flops'], info['train_accs'], info['valid_accs'], info['test_accs'], info['otest_accs']
  print ('{:} collect data done.'.format(time_string()))
  #selected_epochs = [0, 100, 150, 180, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
  selected_epochs = list( range(200) )
  x_xtests = test_accs[199]
  indexes  = list(range(len(x_xtests)))
  ord_idxs = sorted(indexes, key=lambda i: x_xtests[i])
  for sepoch in selected_epochs:
    x_valids = valid_accs[sepoch]
    valid_ord_idxs = sorted(indexes, key=lambda i: x_valids[i])
    valid_ord_lbls = []
    for idx in ord_idxs:
      valid_ord_lbls.append( valid_ord_idxs.index(idx) )
    # labeled data
    dpi, width, height = 300, 2600, 2600
    figsize = width / float(dpi), height / float(dpi)
    LabelSize, LegendFontsize = 18, 18

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    plt.xlim(min(indexes), max(indexes))
    plt.ylim(min(indexes), max(indexes))
    plt.yticks(np.arange(min(indexes), max(indexes), max(indexes)//6), fontsize=LegendFontsize, rotation='vertical')
    plt.xticks(np.arange(min(indexes), max(indexes), max(indexes)//6), fontsize=LegendFontsize)
    ax.scatter(indexes, valid_ord_lbls, marker='^', s=0.5, c='tab:green', alpha=0.8)
    ax.scatter(indexes, indexes       , marker='o', s=0.5, c='tab:blue' , alpha=0.8)
    ax.scatter([-1], [-1], marker='^', s=100, c='tab:green', label='CIFAR-10 validation')
    ax.scatter([-1], [-1], marker='o', s=100, c='tab:blue' , label='CIFAR-10 test')
    plt.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.legend(loc='upper left', fontsize=LegendFontsize)
    ax.set_xlabel('architecture ranking in the final test accuracy', fontsize=LabelSize)
    ax.set_ylabel('architecture ranking in the validation set', fontsize=LabelSize)
    save_path = (vis_save_dir / 'time-{:03d}.pdf'.format(sepoch)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
    save_path = (vis_save_dir / 'time-{:03d}.png'.format(sepoch)).resolve()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
    print ('{:} save into {:}'.format(time_string(), save_path))
    plt.close('all')



def write_video(save_dir):
  import cv2
  video_save_path = save_dir / 'time.avi'
  print ('{:} start create video for {:}'.format(time_string(), video_save_path))
  images = sorted( list( save_dir.glob('time-*.png') ) )
  ximage = cv2.imread(str(images[0]))
  #shape  = (ximage.shape[1], ximage.shape[0])
  shape  = (1000, 1000)
  #writer = cv2.VideoWriter(str(video_save_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, shape)
  writer = cv2.VideoWriter(str(video_save_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, shape)
  for idx, image in enumerate(images):
    ximage = cv2.imread(str(image))
    _image = cv2.resize(ximage, shape)
    writer.write(_image)
  writer.release()
  print ('write video [{:} frames] into {:}'.format(len(images), video_save_path))



if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='NAS-Bench-102', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--save_dir',  type=str, default='./output/search-cell-nas-bench-102/visual', help='The base-name of folder to save checkpoints and log.')
  parser.add_argument('--api_path',  type=str, default=None,                                        help='The path to the NAS-Bench-102 benchmark file.')
  args = parser.parse_args()
  
  vis_save_dir = Path(args.save_dir) / 'visuals'
  vis_save_dir.mkdir(parents=True, exist_ok=True)
  meta_file = Path(args.api_path)
  assert meta_file.exists(), 'invalid path for api : {:}'.format(meta_file)
  visualize_rank_over_time(str(meta_file), vis_save_dir / 'over-time')
  write_video(vis_save_dir / 'over-time')
  visualize_info(str(meta_file), 'cifar10' , vis_save_dir)
  visualize_info(str(meta_file), 'cifar100', vis_save_dir)
  visualize_info(str(meta_file), 'ImageNet16-120', vis_save_dir)
  visualize_relative_ranking(vis_save_dir)
