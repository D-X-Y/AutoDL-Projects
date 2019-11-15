from __future__ import print_function
import numpy as np
from PIL import Image
import pickle as pkl
import os, cv2, csv, glob
import torch
import torch.utils.data as data


class TieredImageNet(data.Dataset):

  def __init__(self, root_dir, split, transform=None):
    self.split = split
    self.root_dir = root_dir
    self.transform = transform
    splits = split.split('-')

    images, labels, last = [], [], 0
    for split in splits:
      labels_name = '{:}/{:}_labels.pkl'.format(self.root_dir, split)
      images_name = '{:}/{:}_images.npz'.format(self.root_dir, split)
      # decompress images if npz not exits
      if not os.path.exists(images_name):
        png_pkl = images_name[:-4] + '_png.pkl'
        if os.path.exists(png_pkl):
          decompress(images_name, png_pkl)
        else:
          raise ValueError('png_pkl {:} not exits'.format( png_pkl ))
      assert os.path.exists(images_name) and os.path.exists(labels_name), '{:} & {:}'.format(images_name, labels_name)
      print ("Prepare {:} done".format(images_name))
      try:
        with open(labels_name) as f:
          data = pkl.load(f)
          label_specific = data["label_specific"]
      except:
        with open(labels_name, 'rb') as f:
          data = pkl.load(f, encoding='bytes')
          label_specific = data[b'label_specific']
      with np.load(images_name, mmap_mode="r", encoding='latin1') as data:
        image_data = data["images"]
      images.append( image_data )
      label_specific = label_specific + last
      labels.append( label_specific )
      last = np.max(label_specific) + 1
      print ("Load {:} done, with image shape = {:}, label shape = {:}, [{:} ~ {:}]".format(images_name, image_data.shape, label_specific.shape, np.min(label_specific), np.max(label_specific)))
    images, labels = np.concatenate(images), np.concatenate(labels)

    self.images = images
    self.labels = labels
    self.n_classes = int( np.max(labels) + 1 )
    self.dict_index_label = {}
    for cls in range(self.n_classes):
      idxs = np.where(labels==cls)[0]
      self.dict_index_label[cls] = idxs
    self.length = len(labels)
    print ("There are {:} images, {:} labels [{:} ~ {:}]".format(images.shape, labels.shape, np.min(labels), np.max(labels)))
  

  def __repr__(self):
    return ('{name}(length={length}, classes={n_classes})'.format(name=self.__class__.__name__, **self.__dict__))

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'invalid index = {:}'.format(index)
    image = self.images[index].copy()
    label = int(self.labels[index])
    image = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
    if self.transform is not None:
      image = self.transform( image )
    return image, label




def decompress(path, output):
  with open(output, 'rb') as f:
    array = pkl.load(f, encoding='bytes')
  images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
  for ii, item in enumerate(array):
    im = cv2.imdecode(item, 1)
    images[ii] = im
  np.savez(path, images=images)
