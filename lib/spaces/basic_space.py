#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.01 #
#####################################################

import abc
import random


class Space(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def random(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError


class Categorical(Space):
    def __init__(self, *data):
        self._candidates = [*data]

    def __getitem__(self, index):
        return self._candidates[index]

    def __len__(self):
        return len(self._candidates)

    def __repr__(self):
        return "{name:}(candidates={cs:})".format(name=self.__class__.__name__, cs=self._candidates)

    def random(self):
        return random.choice(self._candidates)
