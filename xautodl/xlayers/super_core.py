#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
from .super_module import SuperRunMode
from .super_module import IntSpaceType
from .super_module import LayerOrder

from .super_module import SuperModule
from .super_container import SuperSequential
from .super_linear import SuperLinear
from .super_linear import SuperMLPv1, SuperMLPv2

from .super_norm import SuperSimpleNorm
from .super_norm import SuperLayerNorm1D
from .super_norm import SuperSimpleLearnableNorm
from .super_norm import SuperIdentity
from .super_dropout import SuperDropout
from .super_dropout import SuperDrop

super_name2norm = {
    "simple_norm": SuperSimpleNorm,
    "simple_learn_norm": SuperSimpleLearnableNorm,
    "layer_norm_1d": SuperLayerNorm1D,
    "identity": SuperIdentity,
}

from .super_attention import SuperSelfAttention
from .super_attention import SuperQKVAttention
from .super_attention_v2 import SuperQKVAttentionV2
from .super_transformer import SuperTransformerEncoderLayer

from .super_activations import SuperReLU
from .super_activations import SuperLeakyReLU
from .super_activations import SuperTanh
from .super_activations import SuperGELU
from .super_activations import SuperSigmoid

super_name2activation = {
    "relu": SuperReLU,
    "sigmoid": SuperSigmoid,
    "gelu": SuperGELU,
    "leaky_relu": SuperLeakyReLU,
    "tanh": SuperTanh,
}


from .super_trade_stem import SuperAlphaEBDv1
from .super_positional_embedding import SuperDynamicPositionE
from .super_positional_embedding import SuperPositionalEncoder

from .super_rearrange import SuperReArrange
