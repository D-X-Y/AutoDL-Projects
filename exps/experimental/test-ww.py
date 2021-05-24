import sys, time, random, argparse
from copy import deepcopy
import torchvision.models as models
from pathlib import Path

from xautodl.utils import weight_watcher


def main():
    # model = models.vgg19_bn(pretrained=True)
    # _, summary = weight_watcher.analyze(model, alphas=False)
    # for key, value in summary.items():
    #   print('{:10s} : {:}'.format(key, value))

    _, summary = weight_watcher.analyze(models.vgg13(pretrained=True), alphas=False)
    print("vgg-13 : {:}".format(summary["lognorm"]))
    _, summary = weight_watcher.analyze(models.vgg13_bn(pretrained=True), alphas=False)
    print("vgg-13-BN : {:}".format(summary["lognorm"]))
    _, summary = weight_watcher.analyze(models.vgg16(pretrained=True), alphas=False)
    print("vgg-16 : {:}".format(summary["lognorm"]))
    _, summary = weight_watcher.analyze(models.vgg16_bn(pretrained=True), alphas=False)
    print("vgg-16-BN : {:}".format(summary["lognorm"]))
    _, summary = weight_watcher.analyze(models.vgg19(pretrained=True), alphas=False)
    print("vgg-19 : {:}".format(summary["lognorm"]))
    _, summary = weight_watcher.analyze(models.vgg19_bn(pretrained=True), alphas=False)
    print("vgg-19-BN : {:}".format(summary["lognorm"]))


if __name__ == "__main__":
    main()
