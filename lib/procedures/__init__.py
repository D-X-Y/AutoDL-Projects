##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from .starts     import prepare_seed, prepare_logger, get_machine_info, save_checkpoint, copy_checkpoint
from .optimizers import get_optim_scheduler
from .funcs_nasbench import evaluate_for_seed as bench_evaluate_for_seed
from .funcs_nasbench import pure_evaluate as bench_pure_evaluate
from .funcs_nasbench import get_nas_bench_loaders

def get_procedures(procedure):
  from .basic_main     import basic_train, basic_valid
  from .search_main    import search_train, search_valid
  from .search_main_v2 import search_train_v2
  from .simple_KD_main import simple_KD_train, simple_KD_valid

  train_funcs = {'basic' : basic_train, \
                 'search': search_train,'Simple-KD': simple_KD_train, \
                 'search-v2': search_train_v2}
  valid_funcs = {'basic' : basic_valid, \
                 'search': search_valid,'Simple-KD': simple_KD_valid, \
                 'search-v2': search_valid}
  
  train_func  = train_funcs[procedure]
  valid_func  = valid_funcs[procedure]
  return train_func, valid_func
