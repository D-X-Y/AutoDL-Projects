#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################

import os
from pathlib import Path
import abc
import tempfile
import warnings
from typing import Optional, Union, Callable
import torch
import torch.nn as nn
from enum import Enum

from xautodl import spaces
from .super_utils import IntSpaceType, BoolSpaceType
from .super_utils import LayerOrder, SuperRunMode
from .super_utils import TensorContainer
from .super_utils import ShapeContainer

BEST_DIR_KEY = "best_model_dir"
BEST_NAME_KEY = "best_model_name"
BEST_SCORE_KEY = "best_model_score"
ENABLE_CANDIDATE = 0
DISABLE_CANDIDATE = 1


class SuperModule(abc.ABC, nn.Module):
    """This class equips the nn.Module class with the ability to apply AutoDL."""

    def __init__(self):
        super(SuperModule, self).__init__()
        self._super_run_type = SuperRunMode.Default
        self._abstract_child = None
        self._verbose = False
        self._meta_info = {}
        self._candidate_mode = DISABLE_CANDIDATE

    def set_super_run_type(self, super_run_type):
        def _reset_super_run(m):
            if isinstance(m, SuperModule):
                m._super_run_type = super_run_type

        self.apply(_reset_super_run)

    def add_module(self, name: str, module: Optional[torch.nn.Module]) -> None:
        if not isinstance(module, SuperModule):
            warnings.warn(
                "Add {:}:{:} module, which is not SuperModule, into {:}".format(
                    name, module.__class__.__name__, self.__class__.__name__
                )
                + "\n"
                + "It may cause some functions invalid."
            )
        super(SuperModule, self).add_module(name, module)

    def apply_verbose(self, verbose):
        def _reset_verbose(m):
            if isinstance(m, SuperModule):
                m._verbose = verbose

        self.apply(_reset_verbose)

    def apply_candidate(self, abstract_child):
        if not isinstance(abstract_child, spaces.VirtualNode):
            raise ValueError(
                "Invalid abstract child program: {:}".format(abstract_child)
            )
        self._abstract_child = abstract_child

    def enable_candidate(self):
        def _enable_candidate(m):
            if isinstance(m, SuperModule):
                m._candidate_mode = ENABLE_CANDIDATE

        self.apply(_enable_candidate)

    def disable_candidate(self):
        def _disable_candidate(m):
            if isinstance(m, SuperModule):
                m._candidate_mode = DISABLE_CANDIDATE

        self.apply(_disable_candidate)

    def get_w_container(self):
        container = TensorContainer()
        for name, param in self.named_parameters():
            container.append(name, param, True)
        for name, buf in self.named_buffers():
            container.append(name, buf, False)
        return container

    def analyze_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                shapestr = "[{:10s}] shape={:}".format(name, list(param.shape))
                finalstr = shapestr + "{:.2f} +- {:.2f}".format(
                    param.mean(), param.std()
                )
                print(finalstr)

    def numel(self, buffer=True):
        total = 0
        for name, param in self.named_parameters():
            total += param.numel()
        if buffer:
            for name, buf in self.named_buffers():
                total += buf.numel()
        return total

    def set_best_dir(self, xdir):
        self._meta_info[BEST_DIR_KEY] = str(xdir)
        Path(xdir).mkdir(parents=True, exist_ok=True)

    def set_best_name(self, xname):
        self._meta_info[BEST_NAME_KEY] = str(xname)

    def save_best(self, score):
        if BEST_DIR_KEY not in self._meta_info:
            tempdir = tempfile.mkdtemp("-xlayers")
            self._meta_info[BEST_DIR_KEY] = tempdir
        if BEST_SCORE_KEY not in self._meta_info:
            self._meta_info[BEST_SCORE_KEY] = None
        best_score = self._meta_info[BEST_SCORE_KEY]
        if best_score is None or best_score <= score:
            best_save_name = self._meta_info.get(
                BEST_NAME_KEY, "best-{:}.pth".format(self.__class__.__name__)
            )

            best_save_path = os.path.join(self._meta_info[BEST_DIR_KEY], best_save_name)
            self._meta_info[BEST_SCORE_KEY] = score
            torch.save(self.state_dict(), best_save_path)
            return True, self._meta_info[BEST_SCORE_KEY]
        else:
            return False, self._meta_info[BEST_SCORE_KEY]

    def load_best(self, best_save_name=None):
        if BEST_DIR_KEY not in self._meta_info:
            raise ValueError("Please set BEST_DIR_KEY at first")
        if best_save_name is None:
            best_save_name = self._meta_info.get(
                BEST_NAME_KEY, "best-{:}.pth".format(self.__class__.__name__)
            )
        best_save_path = os.path.join(self._meta_info[BEST_DIR_KEY], best_save_name)
        state_dict = torch.load(best_save_path)
        self.load_state_dict(state_dict)

    def has_best(self, best_name=None):
        if BEST_DIR_KEY not in self._meta_info:
            raise ValueError("Please set BEST_DIR_KEY at first")
        if best_name is None:
            best_save_name = self._meta_info.get(
                BEST_NAME_KEY, "best-{:}.pth".format(self.__class__.__name__)
            )
        else:
            best_save_name = best_name
        best_save_path = os.path.join(self._meta_info[BEST_DIR_KEY], best_save_name)
        return os.path.exists(best_save_path)

    @property
    def abstract_search_space(self):
        raise NotImplementedError

    @property
    def super_run_type(self):
        return self._super_run_type

    @property
    def abstract_child(self):
        return self._abstract_child

    @property
    def verbose(self):
        return self._verbose

    @abc.abstractmethod
    def forward_raw(self, *inputs):
        """Use the largest candidate for forward. Similar to the original PyTorch model."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_candidate(self, *inputs):
        raise NotImplementedError

    @property
    def name_with_id(self):
        return "name={:}, id={:}".format(self.__class__.__name__, id(self))

    def get_shape_str(self, tensors):
        if isinstance(tensors, (list, tuple)):
            shapes = [self.get_shape_str(tensor) for tensor in tensors]
            if len(shapes) == 1:
                return shapes[0]
            else:
                return ", ".join(shapes)
        elif isinstance(tensors, (torch.Tensor, nn.Parameter)):
            return str(tuple(tensors.shape))
        else:
            raise TypeError("Invalid input type: {:}.".format(type(tensors)))

    def forward(self, *inputs):
        if self.verbose:
            print(
                "[{:}] inputs shape: {:}".format(
                    self.name_with_id, self.get_shape_str(inputs)
                )
            )
        if self.super_run_type == SuperRunMode.FullModel:
            outputs = self.forward_raw(*inputs)
        elif self.super_run_type == SuperRunMode.Candidate:
            if self._candidate_mode == DISABLE_CANDIDATE:
                raise ValueError("candidate mode is disabled")
            outputs = self.forward_candidate(*inputs)
        else:
            raise ValueError(
                "Unknown Super Model Run Mode: {:}".format(self.super_run_type)
            )
        if self.verbose:
            print(
                "[{:}] outputs shape: {:}".format(
                    self.name_with_id, self.get_shape_str(outputs)
                )
            )
        return outputs

    def forward_with_container(self, inputs, container, prefix=[]):
        raise NotImplementedError
