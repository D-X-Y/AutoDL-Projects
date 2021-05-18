##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# general config related functions
from .config_utils import load_config, dict2config, configure2str

# the args setting for different experiments
from .basic_args import obtain_basic_args
from .attention_args import obtain_attention_args
from .random_baseline import obtain_RandomSearch_args
from .cls_kd_args import obtain_cls_kd_args
from .cls_init_args import obtain_cls_init_args
from .search_single_args import obtain_search_single_args
from .search_args import obtain_search_args

# for network pruning
from .pruning_args import obtain_pruning_args

# utils for args
from .args_utils import arg_str2bool
