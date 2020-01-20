##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from .search_model_gdas     import TinyNetworkGDAS
from .search_model_darts    import TinyNetworkDARTS

nas_super_nets = {'GDAS' : TinyNetworkGDAS,
                  'DARTS': TinyNetworkDARTS}
