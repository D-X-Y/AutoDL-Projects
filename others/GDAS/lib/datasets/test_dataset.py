import os, sys, torch
import torchvision.transforms as transforms

from TieredImageNet import TieredImageNet
from MetaBatchSampler import MetaBatchSampler

root_dir = os.environ['TORCH_HOME'] + '/tiered-imagenet'
print ('root : {:}'.format(root_dir))
means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(84, padding=8), transforms.ToTensor(), transforms.Normalize(means, stds)]
transform = transforms.Compose(lists)

dataset = TieredImageNet(root_dir, 'val-test', transform)
image, label = dataset[111]
print ('image shape = {:}, label = {:}'.format(image.size(), label))
print ('image : min = {:}, max = {:}    ||| label : {:}'.format(image.min(), image.max(), label))


sampler = MetaBatchSampler(dataset.labels, 250, 100, 10)

dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

print ('the length of dataset : {:}'.format( len(dataset) ))
print ('the length of loader  : {:}'.format( len(dataloader) ))

for images, labels in dataloader:
  print ('images : {:}'.format( images.size() ))
  print ('labels : {:}'.format( labels.size() ))
  for i in range(3):
    print ('image-value-[{:}] : {:} ~ {:}, mean={:}, std={:}'.format(i, images[:,i].min(), images[:,i].max(), images[:,i].mean(), images[:,i].std()))

print('-----')
