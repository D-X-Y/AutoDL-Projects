import os, sys, random
from pathlib import Path


def sample_100_cls():
  with open('classes.txt') as f:
    content = f.readlines()
  content = [x.strip() for x in content] 
  random.seed(111)
  classes = random.sample(content, 100)
  classes.sort()
  with open('ImageNet-100.txt', 'w') as f:
    for cls in classes: f.write('{:}\n'.format(cls))
  print('-'*100)


if __name__ == "__main__":
  #sample_100_cls()
  IN1K_root = Path.home() / '.torch' / 'ILSVRC2012'
  IN100_root = Path.home() / '.torch' / 'ILSVRC2012-100'
  assert IN1K_root.exists(), 'ImageNet directory does not exist : {:}'.format(IN1K_root)
  print ('Create soft link from ImageNet directory into : {:}'.format(IN100_root))
  with open('ImageNet-100.txt', 'r') as f:
    classes = f.readlines()
  classes = [x.strip() for x in classes]
  for sub in ['train', 'val']:
    xdir = IN100_root / sub
    if not xdir.exists(): xdir.mkdir(parents=True, exist_ok=True)

  for idx, cls in enumerate(classes):
    xdir = IN1K_root / 'train' / cls
    assert xdir.exists(), '{:} does not exist'.format(xdir)
    os.system('ln -s {:} {:}'.format(xdir, IN100_root / 'train' / cls))

    xdir = IN1K_root / 'val' / cls
    assert xdir.exists(), '{:} does not exist'.format(xdir)
    os.system('ln -s {:} {:}'.format(xdir, IN100_root / 'val' / cls))
