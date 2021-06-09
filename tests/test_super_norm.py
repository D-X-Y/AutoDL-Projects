#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest ./tests/test_super_norm.py -s              #
#####################################################
import unittest

import torch
from xautodl.xlayers import super_core
from xautodl import spaces


class TestSuperSimpleNorm(unittest.TestCase):
    """Test the super simple norm."""

    def test_super_simple_norm(self):
        out_features = spaces.Categorical(12, 24, 36)
        bias = spaces.Categorical(True, False)
        model = super_core.SuperSequential(
            super_core.SuperSimpleNorm(5, 0.5),
            super_core.SuperLinear(10, out_features, bias=bias),
        )
        print("The simple super module is:\n{:}".format(model))
        model.apply_verbose(True)

        print(model.super_run_type)
        self.assertTrue(model[1].bias)

        inputs = torch.rand(20, 10)
        print("Input shape: {:}".format(inputs.shape))
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (20, 36))

        abstract_space = model.abstract_search_space
        abstract_space.clean_last()
        abstract_child = abstract_space.random()
        print("The abstract searc space:\n{:}".format(abstract_space))
        print("The abstract child program:\n{:}".format(abstract_child))

        model.set_super_run_type(super_core.SuperRunMode.Candidate)
        model.enable_candidate()
        model.apply_candidate(abstract_child)

        output_shape = (20, abstract_child["1"]["_out_features"].value)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), output_shape)

    def test_super_simple_learn_norm(self):
        out_features = spaces.Categorical(12, 24, 36)
        bias = spaces.Categorical(True, False)
        model = super_core.SuperSequential(
            super_core.SuperSimpleLearnableNorm(),
            super_core.SuperIdentity(),
            super_core.SuperLinear(10, out_features, bias=bias),
        )
        print("The simple super module is:\n{:}".format(model))
        model.apply_verbose(True)

        print(model.super_run_type)
        self.assertTrue(model[2].bias)

        inputs = torch.rand(20, 10)
        print("Input shape: {:}".format(inputs.shape))
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (20, 36))

        abstract_space = model.abstract_search_space
        abstract_space.clean_last()
        abstract_child = abstract_space.random()
        print("The abstract searc space:\n{:}".format(abstract_space))
        print("The abstract child program:\n{:}".format(abstract_child))

        model.set_super_run_type(super_core.SuperRunMode.Candidate)
        model.enable_candidate()
        model.apply_candidate(abstract_child)

        output_shape = (20, abstract_child["2"]["_out_features"].value)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), output_shape)
