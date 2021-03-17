import sys, time, random, argparse
from copy import deepcopy
import torchvision.models as models
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from utils import get_model_infos

# from models.ImageNet_MobileNetV2 import MobileNetV2
from torchvision.models.mobilenet import MobileNetV2


def main(width_mult):
    # model = MobileNetV2(1001, width_mult, 32, 1280, 'InvertedResidual', 0.2)
    model = MobileNetV2(width_mult=width_mult)
    print(model)
    flops, params = get_model_infos(model, (2, 3, 224, 224))
    print("FLOPs : {:}".format(flops))
    print("Params : {:}".format(params))
    print("-" * 50)


if __name__ == "__main__":
    main(1.0)
    main(1.4)
