from .model_search    import Network
from .model_search_v1 import NetworkV1
from .model_search_f1 import NetworkF1
# acceleration model
from .model_search_f1_acc2 import NetworkFACC1
from .model_search_acc2 import NetworkACC2
from .model_search_v3 import NetworkV3
from .model_search_v4 import NetworkV4
from .model_search_v5 import NetworkV5
from .CifarNet import NetworkCIFAR
from .ImageNet import NetworkImageNet

# genotypes
from .genotypes import DARTS_V1, DARTS_V2
from .genotypes import NASNet, PNASNet, AmoebaNet, ENASNet
from .genotypes import DMS_V1, DMS_F1, GDAS_CC

from .construct_utils import return_alphas_str
