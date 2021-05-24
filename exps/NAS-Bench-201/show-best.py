#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.01 #
################################################################################################
# python exps/NAS-Bench-201/show-best.py --api_path $HOME/.torch/NAS-Bench-201-v1_0-e61699.pth #
################################################################################################
import argparse
from pathlib import Path

from nas_201_api import NASBench201API as API

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis of NAS-Bench-201")
    parser.add_argument(
        "--api_path",
        type=str,
        default=None,
        help="The path to the NAS-Bench-201 benchmark file.",
    )
    args = parser.parse_args()

    meta_file = Path(args.api_path)
    assert meta_file.exists(), "invalid path for api : {:}".format(meta_file)

    api = API(str(meta_file))

    # This will show the results of the best architecture based on the validation set of each dataset.
    arch_index, accuracy = api.find_best("cifar10-valid", "x-valid", None, None, False)
    print("FOR CIFAR-010, using the hyper-parameters with 200 training epochs :::")
    print("arch-index={:5d}, arch={:}".format(arch_index, api.arch(arch_index)))
    api.show(arch_index)
    print("")

    arch_index, accuracy = api.find_best("cifar100", "x-valid", None, None, False)
    print("FOR CIFAR-100, using the hyper-parameters with 200 training epochs :::")
    print("arch-index={:5d}, arch={:}".format(arch_index, api.arch(arch_index)))
    api.show(arch_index)
    print("")

    arch_index, accuracy = api.find_best("ImageNet16-120", "x-valid", None, None, False)
    print("FOR ImageNet16-120, using the hyper-parameters with 200 training epochs :::")
    print("arch-index={:5d}, arch={:}".format(arch_index, api.arch(arch_index)))
    api.show(arch_index)
    print("")
