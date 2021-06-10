#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest ./tests/test_super_att.py -s               #
#####################################################
import random
import unittest
from parameterized import parameterized

import torch
from xautodl import spaces
from xautodl.xlayers import super_core


class TestSuperSelfAttention(unittest.TestCase):
    """Test the super attention layer."""

    def _internal_func(self, inputs, model):
        outputs = model(inputs)
        abstract_space = model.abstract_search_space
        print(
            "The abstract search space for SuperSelfAttention is:\n{:}".format(
                abstract_space
            )
        )
        abstract_space.clean_last()
        abstract_child = abstract_space.random(reuse_last=True)
        print("The abstract child program is:\n{:}".format(abstract_child))
        model.set_super_run_type(super_core.SuperRunMode.Candidate)
        model.enable_candidate()
        model.apply_candidate(abstract_child)
        outputs = model(inputs)
        return abstract_child, outputs

    def test_super_attention(self):
        proj_dim = spaces.Categorical(12, 24, 36)
        num_heads = spaces.Categorical(2, 4, 6)
        model = super_core.SuperSelfAttention(10, proj_dim, num_heads)
        print(model)
        model.apply_verbose(True)

        inputs = torch.rand(4, 20, 10)  # batch size, sequence length, channel
        abstract_child, outputs = self._internal_func(inputs, model)
        output_shape = (4, 20, abstract_child["proj"]["_out_features"].value)
        self.assertEqual(tuple(outputs.shape), output_shape)

    @parameterized.expand([[6], [12], [24], [48]])
    def test_transformer_encoder(self, input_dim):
        output_dim = spaces.Categorical(12, 24, 36)
        model = super_core.SuperSequential(
            super_core.SuperLinear(input_dim, output_dim),
            super_core.SuperTransformerEncoderLayer(
                output_dim,
                num_heads=spaces.Categorical(2, 4, 6),
                mlp_hidden_multiplier=spaces.Categorical(1, 2, 4),
            ),
        )
        print(model)
        model.apply_verbose(True)
        inputs = torch.rand(4, 20, input_dim)
        abstract_child, outputs = self._internal_func(inputs, model)
        output_shape = (
            4,
            20,
            output_dim.abstract(reuse_last=True).random(reuse_last=True).value,
        )
        self.assertEqual(tuple(outputs.shape), output_shape)
