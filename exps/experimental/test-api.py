# 
# exps/experimental/test-api.py
#
import sys, time, random, argparse
from copy import deepcopy
import torchvision.models as models
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from nas_201_api import NASBench201API as API


def main():
  api = API(None)
  info = api.get_more_info(100, 'cifar100', 199, False, True)


if __name__ == '__main__':
  main()
