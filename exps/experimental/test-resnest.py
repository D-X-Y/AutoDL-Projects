#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06 #
#####################################################
# python exps/experimental/test-resnest.py
#####################################################
import sys, time, torch, random, argparse
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from utils import get_model_infos

torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)

for model_name, xshape in [
    ("resnest50", (1, 3, 224, 224)),
    ("resnest101", (1, 3, 256, 256)),
    ("resnest200", (1, 3, 320, 320)),
    ("resnest269", (1, 3, 416, 416)),
]:
    # net = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=True)
    net = torch.hub.load("zhanghang1989/ResNeSt", model_name, pretrained=False)
    print("Model : {:}, input shape : {:}".format(model_name, xshape))
    flops, param = get_model_infos(net, xshape)
    print("flops  : {:.3f}M".format(flops))
    print("params : {:.3f}M".format(param))
