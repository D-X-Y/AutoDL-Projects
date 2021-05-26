#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
# I write this package to make AutoDL-Projects to be compatible with the old GDAS projects.
# Ideally, this package will be merged into lib/models/cell_infers in future.
# Currently, this package is used to reproduce the results in GDAS (Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019).
##################################################

import os, torch


def obtain_nas_infer_model(config, extra_model_path=None):

    if config.arch == "dxys":
        from .DXYs import CifarNet, ImageNet, Networks
        from .DXYs import build_genotype_from_dict

        if config.genotype is None:
            if extra_model_path is not None and not os.path.isfile(extra_model_path):
                raise ValueError(
                    "When genotype in confiig is None, extra_model_path must be set as a path instead of {:}".format(
                        extra_model_path
                    )
                )
            xdata = torch.load(extra_model_path)
            current_epoch = xdata["epoch"]
            genotype_dict = xdata["genotypes"][current_epoch - 1]
            genotype = build_genotype_from_dict(genotype_dict)
        else:
            genotype = Networks[config.genotype]
        if config.dataset == "cifar":
            return CifarNet(
                config.ichannel,
                config.layers,
                config.stem_multi,
                config.auxiliary,
                genotype,
                config.class_num,
            )
        elif config.dataset == "imagenet":
            return ImageNet(
                config.ichannel,
                config.layers,
                config.auxiliary,
                genotype,
                config.class_num,
            )
        else:
            raise ValueError("invalid dataset : {:}".format(config.dataset))
    else:
        raise ValueError("invalid nas arch type : {:}".format(config.arch))
