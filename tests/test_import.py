#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest ./tests/test_import.py                     #
#####################################################
def test_import():
    from xautodl import config_utils
    from xautodl import datasets
    from xautodl import log_utils
    from xautodl import models
    from xautodl import nas_infer_model
    from xautodl import procedures
    from xautodl import trade_models
    from xautodl import utils

    from xautodl import xlayers
    from xautodl import xmisc
    from xautodl import xmodels
    from xautodl import spaces

    print("Check all imports done")
