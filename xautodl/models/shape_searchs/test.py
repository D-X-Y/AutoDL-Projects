##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
import torch.nn as nn
from SoftSelect import ChannelWiseInter


if __name__ == "__main__":

    tensors = torch.rand((16, 128, 7, 7))

    for oc in range(200, 210):
        out_v1 = ChannelWiseInter(tensors, oc, "v1")
        out_v2 = ChannelWiseInter(tensors, oc, "v2")
        assert (out_v1 == out_v2).any().item() == 1
    for oc in range(48, 160):
        out_v1 = ChannelWiseInter(tensors, oc, "v1")
        out_v2 = ChannelWiseInter(tensors, oc, "v2")
        assert (out_v1 == out_v2).any().item() == 1
