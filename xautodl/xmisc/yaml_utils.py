#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.06 #
#####################################################
import os
import yaml


def load_yaml(path):
    if not os.path.isfile(path):
        raise ValueError("{:} is not a file.".format(path))
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return data
