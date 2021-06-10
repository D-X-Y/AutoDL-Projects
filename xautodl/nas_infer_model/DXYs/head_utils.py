import torch
import torch.nn as nn


class ImageNetHEAD(nn.Sequential):
    def __init__(self, C, stride=2):
        super(ImageNetHEAD, self).__init__()
        self.add_module(
            "conv1",
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.add_module("bn1", nn.BatchNorm2d(C // 2))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv2",
            nn.Conv2d(C // 2, C, kernel_size=3, stride=stride, padding=1, bias=False),
        )
        self.add_module("bn2", nn.BatchNorm2d(C))


class CifarHEAD(nn.Sequential):
    def __init__(self, C):
        super(CifarHEAD, self).__init__()
        self.add_module("conv", nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False))
        self.add_module("bn", nn.BatchNorm2d(C))


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(
                5, stride=3, padding=0, count_include_pad=False
            ),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
