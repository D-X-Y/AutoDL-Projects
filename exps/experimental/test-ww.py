import sys, time, random, argparse
from copy import deepcopy
import torchvision.models as models
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from utils import weight_watcher


def main():
  model = models.vgg19_bn(pretrained=True)
  _, summary = weight_watcher.analyze(model, alphas=False)
  # print(summary)
  for key, value in summary.items():
    print('{:10s} : {:}'.format(key, value))
  # import pdb; pdb.set_trace()


if __name__ == '__main__':
  main()