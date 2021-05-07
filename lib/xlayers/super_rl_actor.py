#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# DISABLED / NOT-FINISHED
#####################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable

import spaces
from .super_container import SuperSequential
from .super_linear import SuperLinear


class SuperActor(SuperModule):
    """A Actor in RL."""

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward_candidate(self, **kwargs):
        return self.forward_raw(**kwargs)

    def forward_raw(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class SuperLfnaMetaMLP(SuperModule):
    def __init__(self, obs_dim, hidden_sizes, act_cls):
        super(SuperLfnaMetaMLP).__init__()
        self.delta_net = SuperSequential(
            SuperLinear(obs_dim, hidden_sizes[0]),
            act_cls(),
            SuperLinear(hidden_sizes[0], hidden_sizes[1]),
            act_cls(),
            SuperLinear(hidden_sizes[1], 1),
        )


class SuperLfnaMetaMLP(SuperModule):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_cls):
        super(SuperLfnaMetaMLP).__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = SuperSequential(
            SuperLinear(obs_dim, hidden_sizes[0]),
            act_cls(),
            SuperLinear(hidden_sizes[0], hidden_sizes[1]),
            act_cls(),
            SuperLinear(hidden_sizes[1], act_dim),
        )

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward_candidate(self, **kwargs):
        return self.forward_raw(**kwargs)

    def forward_raw(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class SuperMLPGaussianActor(SuperModule):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_cls):
        super(SuperMLPGaussianActor).__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = SuperSequential(
            SuperLinear(obs_dim, hidden_sizes[0]),
            act_cls(),
            SuperLinear(hidden_sizes[0], hidden_sizes[1]),
            act_cls(),
            SuperLinear(hidden_sizes[1], act_dim),
        )

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward_candidate(self, **kwargs):
        return self.forward_raw(**kwargs)

    def forward_raw(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
