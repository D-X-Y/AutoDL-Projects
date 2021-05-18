##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os


def test_imagenet_data(imagenet):
    total_length = len(imagenet)
    assert (
        total_length == 1281166 or total_length == 50000
    ), "The length of ImageNet is wrong : {}".format(total_length)
    map_id = {}
    for index in range(total_length):
        path, target = imagenet.imgs[index]
        folder, image_name = os.path.split(path)
        _, folder = os.path.split(folder)
        if folder not in map_id:
            map_id[folder] = target
        else:
            assert map_id[folder] == target, "Class : {} is not {}".format(
                folder, target
            )
        assert image_name.find(folder) == 0, "{} is wrong.".format(path)
    print("Check ImageNet Dataset OK")
