#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import torch

from itertools import islice
import operator

from collections import OrderedDict
from typing import Optional, Union, Callable, TypeVar, Iterator

from xautodl import spaces
from .super_module import SuperModule


T = TypeVar("T", bound=SuperModule)


class SuperSequential(SuperModule):
    """A sequential container wrapped with 'Super' ability.

    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    To make it easier to understand, here is a small example::
        # Example of using Sequential
        model = SuperSequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(SuperSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            if not isinstance(args, (list, tuple)):
                raise ValueError("Invalid input type: {:}".format(type(args)))
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError("index {} is out of range".format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx) -> Union["SuperSequential", T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: SuperModule) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        return len(self._modules)

    def __dir__(self):
        keys = super(SuperSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[SuperModule]:
        return iter(self._modules.values())

    @property
    def abstract_search_space(self):
        root_node = spaces.VirtualNode(id(self))
        for index, module in enumerate(self):
            if not isinstance(module, SuperModule):
                continue
            space = module.abstract_search_space
            if not spaces.is_determined(space):
                root_node.append(str(index), space)
        return root_node

    def apply_candidate(self, abstract_child: spaces.VirtualNode):
        super(SuperSequential, self).apply_candidate(abstract_child)
        for index, module in enumerate(self):
            if str(index) in abstract_child:
                module.apply_candidate(abstract_child[str(index)])

    def forward_candidate(self, input):
        return self.forward_raw(input)

    def forward_raw(self, input):
        for module in self:
            input = module(input)
        return input

    def forward_with_container(self, input, container, prefix=[]):
        for index, module in enumerate(self):
            input = module.forward_with_container(
                input, container, prefix + [str(index)]
            )
        return input
