#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
########################################################
# python exps/NAS-Bench-201/test-correlation.py --api_path $HOME/.torch/NAS-Bench-201-v1_0-e61699.pth
########################################################
import sys, argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
from pathlib import Path

from xautodl.log_utils import time_string
from xautodl.models import CellStructure
from nas_201_api import NASBench201API as API


def check_unique_arch(meta_file):
    api = API(str(meta_file))
    arch_strs = deepcopy(api.meta_archs)
    xarchs = [CellStructure.str2structure(x) for x in arch_strs]

    def get_unique_matrix(archs, consider_zero):
        UniquStrs = [arch.to_unique_str(consider_zero) for arch in archs]
        print(
            "{:} create unique-string ({:}/{:}) done".format(
                time_string(), len(set(UniquStrs)), len(UniquStrs)
            )
        )
        Unique2Index = dict()
        for index, xstr in enumerate(UniquStrs):
            if xstr not in Unique2Index:
                Unique2Index[xstr] = list()
            Unique2Index[xstr].append(index)
        sm_matrix = torch.eye(len(archs)).bool()
        for _, xlist in Unique2Index.items():
            for i in xlist:
                for j in xlist:
                    sm_matrix[i, j] = True
        unique_ids, unique_num = [-1 for _ in archs], 0
        for i in range(len(unique_ids)):
            if unique_ids[i] > -1:
                continue
            neighbours = sm_matrix[i].nonzero().view(-1).tolist()
            for nghb in neighbours:
                assert unique_ids[nghb] == -1, "impossible"
                unique_ids[nghb] = unique_num
            unique_num += 1
        return sm_matrix, unique_ids, unique_num

    print(
        "There are {:} valid-archs".format(sum(arch.check_valid() for arch in xarchs))
    )
    sm_matrix, uniqueIDs, unique_num = get_unique_matrix(xarchs, None)
    print(
        "{:} There are {:} unique architectures (considering nothing).".format(
            time_string(), unique_num
        )
    )
    sm_matrix, uniqueIDs, unique_num = get_unique_matrix(xarchs, False)
    print(
        "{:} There are {:} unique architectures (not considering zero).".format(
            time_string(), unique_num
        )
    )
    sm_matrix, uniqueIDs, unique_num = get_unique_matrix(xarchs, True)
    print(
        "{:} There are {:} unique architectures (considering zero).".format(
            time_string(), unique_num
        )
    )


def check_cor_for_bandit(
    meta_file, test_epoch, use_less_or_not, is_rand=True, need_print=False
):
    if isinstance(meta_file, API):
        api = meta_file
    else:
        api = API(str(meta_file))
    cifar10_currs = []
    cifar10_valid = []
    cifar10_test = []
    cifar100_valid = []
    cifar100_test = []
    imagenet_test = []
    imagenet_valid = []
    for idx, arch in enumerate(api):
        results = api.get_more_info(
            idx, "cifar10-valid", test_epoch - 1, use_less_or_not, is_rand
        )
        cifar10_currs.append(results["valid-accuracy"])
        # --->>>>>
        results = api.get_more_info(idx, "cifar10-valid", None, False, is_rand)
        cifar10_valid.append(results["valid-accuracy"])
        results = api.get_more_info(idx, "cifar10", None, False, is_rand)
        cifar10_test.append(results["test-accuracy"])
        results = api.get_more_info(idx, "cifar100", None, False, is_rand)
        cifar100_test.append(results["test-accuracy"])
        cifar100_valid.append(results["valid-accuracy"])
        results = api.get_more_info(idx, "ImageNet16-120", None, False, is_rand)
        imagenet_test.append(results["test-accuracy"])
        imagenet_valid.append(results["valid-accuracy"])

    def get_cor(A, B):
        return float(np.corrcoef(A, B)[0, 1])

    cors = []
    for basestr, xlist in zip(
        ["C-010-V", "C-010-T", "C-100-V", "C-100-T", "I16-V", "I16-T"],
        [
            cifar10_valid,
            cifar10_test,
            cifar100_valid,
            cifar100_test,
            imagenet_valid,
            imagenet_test,
        ],
    ):
        correlation = get_cor(cifar10_currs, xlist)
        if need_print:
            print(
                "With {:3d}/{:}-epochs-training, the correlation between cifar10-valid and {:} is : {:}".format(
                    test_epoch,
                    "012" if use_less_or_not else "200",
                    basestr,
                    correlation,
                )
            )
        cors.append(correlation)
        # print ('With {:3d}/200-epochs-training, the correlation between cifar10-valid and {:} is : {:}'.format(test_epoch, basestr, get_cor(cifar10_valid_200, xlist)))
        # print('-'*200)
    # print('*'*230)
    return cors


def check_cor_for_bandit_v2(meta_file, test_epoch, use_less_or_not, is_rand):
    corrs = []
    for i in tqdm(range(100)):
        x = check_cor_for_bandit(meta_file, test_epoch, use_less_or_not, is_rand, False)
        corrs.append(x)
    # xstrs = ['CIFAR-010', 'C-100-V', 'C-100-T', 'I16-V', 'I16-T']
    xstrs = ["C-010-V", "C-010-T", "C-100-V", "C-100-T", "I16-V", "I16-T"]
    correlations = np.array(corrs)
    print(
        "------>>>>>>>> {:03d}/{:} >>>>>>>> ------".format(
            test_epoch, "012" if use_less_or_not else "200"
        )
    )
    for idx, xstr in enumerate(xstrs):
        print(
            "{:8s} ::: mean={:.4f}, std={:.4f} :: {:.4f}\\pm{:.4f}".format(
                xstr,
                correlations[:, idx].mean(),
                correlations[:, idx].std(),
                correlations[:, idx].mean(),
                correlations[:, idx].std(),
            )
        )
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis of NAS-Bench-201")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/search-cell-nas-bench-201/visuals",
        help="The base-name of folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--api_path",
        type=str,
        default=None,
        help="The path to the NAS-Bench-201 benchmark file.",
    )
    args = parser.parse_args()

    vis_save_dir = Path(args.save_dir)
    vis_save_dir.mkdir(parents=True, exist_ok=True)
    meta_file = Path(args.api_path)
    assert meta_file.exists(), "invalid path for api : {:}".format(meta_file)

    # check_unique_arch(meta_file)
    api = API(str(meta_file))
    # for iepoch in [11, 25, 50, 100, 150, 175, 200]:
    #  check_cor_for_bandit(api,  6, iepoch)
    #  check_cor_for_bandit(api, 12, iepoch)
    check_cor_for_bandit_v2(api, 6, True, True)
    check_cor_for_bandit_v2(api, 12, True, True)
    check_cor_for_bandit_v2(api, 12, False, True)
    check_cor_for_bandit_v2(api, 24, False, True)
    check_cor_for_bandit_v2(api, 100, False, True)
    check_cor_for_bandit_v2(api, 150, False, True)
    check_cor_for_bandit_v2(api, 175, False, True)
    check_cor_for_bandit_v2(api, 200, False, True)
    print("----")
