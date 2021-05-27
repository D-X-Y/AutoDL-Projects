#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
# This directory contains some ad-hoc functions, classes, etc.
# It will be re-formulated in the future.
#####################################################
from .evaluation_utils import obtain_accuracy
from .gpu_manager import GPUManager
from .flop_benchmark import get_model_infos, count_parameters, count_parameters_in_MB
from .affine_utils import normalize_points, denormalize_points
from .affine_utils import identity2affine, solve2theta, affine2image
from .hash_utils import get_md5_file
from .str_utils import split_str2indexes
from .str_utils import show_mean_var
