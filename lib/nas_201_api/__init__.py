#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################################
# This API will not be updated after 2020.09.16.                    #
# Please use our new API in NATS-Bench, which is                    #
# more efficient and contains info of more architecture candidates. #
#####################################################################
from .api_utils import ArchResults, ResultsCount
from .api_201 import NASBench201API

# NAS_BENCH_201_API_VERSION="v1.1"  # [2020.02.25]
# NAS_BENCH_201_API_VERSION="v1.2"  # [2020.03.09]
# NAS_BENCH_201_API_VERSION="v1.3"  # [2020.03.16]
NAS_BENCH_201_API_VERSION="v2.0"    # [2020.06.30]

