import os, sys
from pathlib import Path

url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def load_val():
  path = 'tiny-imagenet-200/val/val_annotations.txt'
  cfile = open(path, 'r')
  content = cfile.readlines()
  content = [x.strip().split('\t') for x in content]
  cfile.close()
  images = [x[0] for x in content]
  labels = [x[1] for x in content]
  return images, labels

def main():
  os.system("wget {:}".format(url))
  os.system("rm -rf tiny-imagenet-200")
  os.system("unzip -o tiny-imagenet-200.zip")
  images, labels = load_val()
  savedir = 'tiny-imagenet-200/new_val'
  if not os.path.exists(savedir): os.makedirs(savedir)
  for image, label in zip(images, labels):
    cdir = savedir + '/' + label
    if not os.path.exists(cdir): os.makedirs(cdir)
    ori_path = 'tiny-imagenet-200/val/images/' + image
    os.system("cp {:} {:}".format(ori_path, cdir))
  os.system("rm -rf tiny-imagenet-200/val")
  os.system("mv {:} tiny-imagenet-200/val".format(savedir))

def generate_salt_pepper():
  targetdir = Path('tiny-imagenet-200/val')
  noisedir  = Path('tiny-imagenet-200/val-noise')
  assert targetdir.exists(), '{:} does not exist'.format(targetdir)
  from imgaug import augmenters as iaa
  import cv2
  aug = iaa.SaltAndPepper(p=0.2)

  for sub in targetdir.iterdir():
    if not sub.is_dir(): continue
    subdir = noisedir / sub.name
    if not subdir.exists(): os.makedirs('{:}'.format(subdir))
    images = sub.glob('*.JPEG')
    for image in images:
      I = cv2.imread(str(image))
      Inoise = aug.augment_image(I)
      savepath = subdir / image.name
      cv2.imwrite(str(savepath), Inoise)
    print ('{:} done'.format(sub))

if __name__ == "__main__":
  #main()
  generate_salt_pepper()
