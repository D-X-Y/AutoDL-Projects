import os, sys, torch
import os.path as osp
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from utils import Cutout
from .TieredImageNet import TieredImageNet

Dataset2Class = {'cifar10' : 10,
                 'cifar100': 100,
                 'tiered'  : -1,
                 'imagnet-1k'  : 1000,
                 'imagenet-100': 100}


def get_datasets(name, root, cutout):

  # Mean + Std
  if name == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif name == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
  elif name == 'tiered':
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  elif name == 'imagnet-1k' or name == 'imagenet-100':
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  else: raise TypeError("Unknow dataset : {:}".format(name))


  # Data Argumentation
  if name == 'cifar10' or name == 'cifar100':
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean, std)]
    if cutout > 0 : lists += [Cutout(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
  elif name == 'tiered':
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(80, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
    if cutout > 0 : lists += [Cutout(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.CenterCrop(80), transforms.ToTensor(), transforms.Normalize(mean, std)])
  elif name == 'imagnet-1k' or name == 'imagenet-100':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ])
    test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
  else: raise TypeError("Unknow dataset : {:}".format(name))
    train_data = TieredImageNet(root, 'train-val', train_transform)
    test_data = None
  if name == 'cifar10':
    train_data = dset.CIFAR10(root, train=True, transform=train_transform, download=True)
    test_data  = dset.CIFAR10(root, train=True, transform=test_transform , download=True)
  elif name == 'cifar100':
    train_data = dset.CIFAR100(root, train=True, transform=train_transform, download=True)
    test_data  = dset.CIFAR100(root, train=True, transform=test_transform , download=True)
  elif name == 'imagnet-1k' or name == 'imagenet-100':
    train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
    test_data  = dset.ImageFolder(osp.join(root, 'val'), train_transform)
  else: raise TypeError("Unknow dataset : {:}".format(name))
  
  class_num = Dataset2Class[name]
  return train_data, test_data, class_num
