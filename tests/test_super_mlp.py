#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest ./tests/test_super_model.py -s             #
#####################################################
import torch
import unittest

from xautodl.xlayers import super_core
from xautodl import spaces


class TestSuperLinear(unittest.TestCase):
    """Test the super linear."""

    def test_super_linear(self):
        out_features = spaces.Categorical(12, 24, 36)
        bias = spaces.Categorical(True, False)
        model = super_core.SuperLinear(10, out_features, bias=bias)
        print("The simple super linear module is:\n{:}".format(model))
        model.apply_verbose(True)

        print(model.super_run_type)
        self.assertTrue(model.bias)

        inputs = torch.rand(20, 10)
        print("Input shape: {:}".format(inputs.shape))
        print("Weight shape: {:}".format(model._super_weight.shape))
        print("Bias shape: {:}".format(model._super_bias.shape))
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

        output_shape = (20, abstract_child["_out_features"].value)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), output_shape)

    def test_super_mlp_v1(self):
        hidden_features = spaces.Categorical(12, 24, 36)
        out_features = spaces.Categorical(24, 36, 48)
        mlp = super_core.SuperMLPv1(10, hidden_features, out_features)
        print(mlp)
        mlp.apply_verbose(False)
        self.assertTrue(mlp.fc1._out_features, mlp.fc2._in_features)

        inputs = torch.rand(4, 10)
        outputs = mlp(inputs)
        self.assertEqual(tuple(outputs.shape), (4, 48))

        abstract_space = mlp.abstract_search_space
        print(
            "The abstract search space for SuperMLPv1 is:\n{:}".format(abstract_space)
        )
        self.assertEqual(
            abstract_space["fc1"]["_out_features"],
            abstract_space["fc2"]["_in_features"],
        )
        self.assertTrue(
            abstract_space["fc1"]["_out_features"]
            is abstract_space["fc2"]["_in_features"]
        )

        abstract_space.clean_last()
        abstract_child = abstract_space.random(reuse_last=True)
        print("The abstract child program is:\n{:}".format(abstract_child))
        self.assertEqual(
            abstract_child["fc1"]["_out_features"].value,
            abstract_child["fc2"]["_in_features"].value,
        )

        mlp.set_super_run_type(super_core.SuperRunMode.Candidate)
        mlp.enable_candidate()
        mlp.apply_candidate(abstract_child)
        outputs = mlp(inputs)
        output_shape = (4, abstract_child["fc2"]["_out_features"].value)
        self.assertEqual(tuple(outputs.shape), output_shape)

    def test_super_mlp_v2(self):
        hidden_multiplier = spaces.Categorical(1.0, 2.0, 3.0)
        out_features = spaces.Categorical(24, 36, 48)
        mlp = super_core.SuperMLPv2(10, hidden_multiplier, out_features)
        print(mlp)
        mlp.apply_verbose(False)

        inputs = torch.rand(4, 10)
        outputs = mlp(inputs)
        self.assertEqual(tuple(outputs.shape), (4, 48))

        abstract_space = mlp.abstract_search_space
        print(
            "The abstract search space for SuperMLPv2 is:\n{:}".format(abstract_space)
        )

        abstract_space.clean_last()
        abstract_child = abstract_space.random(reuse_last=True)
        print("The abstract child program is:\n{:}".format(abstract_child))

        mlp.set_super_run_type(super_core.SuperRunMode.Candidate)
        mlp.enable_candidate()
        mlp.apply_candidate(abstract_child)
        outputs = mlp(inputs)
        output_shape = (4, abstract_child["_out_features"].value)
        self.assertEqual(tuple(outputs.shape), output_shape)

    def test_super_stem(self):
        out_features = spaces.Categorical(24, 36, 48)
        model = super_core.SuperAlphaEBDv1(6, out_features)
        inputs = torch.rand(4, 360)

        abstract_space = model.abstract_search_space
        abstract_space.clean_last()
        abstract_child = abstract_space.random(reuse_last=True)
        print("The abstract searc space:\n{:}".format(abstract_space))
        print("The abstract child program:\n{:}".format(abstract_child))

        model.set_super_run_type(super_core.SuperRunMode.Candidate)
        model.enable_candidate()
        model.apply_candidate(abstract_child)
        outputs = model(inputs)
        output_shape = (4, 60, abstract_child["_embed_dim"].value)
        self.assertEqual(tuple(outputs.shape), output_shape)
