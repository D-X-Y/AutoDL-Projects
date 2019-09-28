import random, tarfile
import numpy, six
from six.moves import cPickle as pickle
from PIL import Image, ImageOps


def train_cifar_augmentation(image):
  # flip
  if random.random() < 0.5: image1 = image.transpose(Image.FLIP_LEFT_RIGHT)
  else:                     image1 = image
  # random crop
  image2 = ImageOps.expand(image1, border=4, fill=0)
  i = random.randint(0, 40 - 32)
  j = random.randint(0, 40 - 32)
  image3 = image2.crop((j,i,j+32,i+32))
  # to numpy
  image3 = numpy.array(image3) / 255.0
  mean   = numpy.array([x / 255 for x in [125.3, 123.0, 113.9]]).reshape(1, 1, 3)
  std    = numpy.array([x / 255 for x in [63.0, 62.1, 66.7]]).reshape(1, 1, 3)
  return (image3 - mean) / std


def valid_cifar_augmentation(image):
  image3 = numpy.array(image) / 255.0
  mean   = numpy.array([x / 255 for x in [125.3, 123.0, 113.9]]).reshape(1, 1, 3)
  std    = numpy.array([x / 255 for x in [63.0, 62.1, 66.7]]).reshape(1, 1, 3)
  return (image3 - mean) / std


def reader_creator(filename, sub_name, is_train, cycle=False):
  def read_batch(batch):
    data = batch[six.b('data')]
    labels = batch.get(
      six.b('labels'), batch.get(six.b('fine_labels'), None))
    assert labels is not None
    for sample, label in six.moves.zip(data, labels):
      sample = sample.reshape(3, 32, 32)
      sample = sample.transpose((1, 2, 0))
      image  = Image.fromarray(sample)
      if is_train:
        ximage = train_cifar_augmentation(image)
      else:
        ximage = valid_cifar_augmentation(image)
      ximage = ximage.transpose((2, 0, 1))
      yield ximage.astype(numpy.float32), int(label)

  def reader():
    with tarfile.open(filename, mode='r') as f:
      names = (each_item.name for each_item in f
           if sub_name in each_item.name)

      while True:
        for name in names:
          if six.PY2:
            batch = pickle.load(f.extractfile(name))
          else:
            batch = pickle.load(
              f.extractfile(name), encoding='bytes')
          for item in read_batch(batch):
            yield item
        if not cycle:
          break

  return reader
