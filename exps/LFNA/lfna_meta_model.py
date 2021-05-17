#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
import copy
import torch

import torch.nn.functional as F

from xlayers import super_core
from xlayers import trunc_normal_
from models.xcore import get_model


class LFNA_Meta(super_core.SuperModule):
    """Learning to Forecast Neural Adaptation (Meta Model Design)."""

    def __init__(
        self,
        shape_container,
        layer_embedding,
        time_embedding,
        meta_timestamps,
        mha_depth: int = 2,
        dropout: float = 0.1,
    ):
        super(LFNA_Meta, self).__init__()
        self._shape_container = shape_container
        self._num_layers = len(shape_container)
        self._numel_per_layer = []
        for ilayer in range(self._num_layers):
            self._numel_per_layer.append(shape_container[ilayer].numel())
        self._raw_meta_timestamps = meta_timestamps

        self.register_parameter(
            "_super_layer_embed",
            torch.nn.Parameter(torch.Tensor(self._num_layers, layer_embedding)),
        )
        self.register_parameter(
            "_super_meta_embed",
            torch.nn.Parameter(torch.Tensor(len(meta_timestamps), time_embedding)),
        )
        self.register_buffer("_meta_timestamps", torch.Tensor(meta_timestamps))
        self._time_embed_dim = time_embedding
        self._append_meta_embed = dict(fixed=None, learnt=None)
        self._append_meta_timestamps = dict(fixed=None, learnt=None)

        # build transformer
        layers = []
        for ilayer in range(mha_depth):
            layers.append(
                super_core.SuperTransformerEncoderLayer(
                    time_embedding,
                    4,
                    True,
                    4,
                    dropout,
                    norm_affine=False,
                    order=super_core.LayerOrder.PostNorm,
                )
            )
        layers.append(super_core.SuperLinear(time_embedding, time_embedding))
        self.meta_corrector = super_core.SuperSequential(*layers)

        model_kwargs = dict(
            config=dict(model_type="dual_norm_mlp"),
            input_dim=layer_embedding + time_embedding,
            output_dim=max(self._numel_per_layer),
            hidden_dims=[(layer_embedding + time_embedding) * 2] * 3,
            act_cls="gelu",
            norm_cls="layer_norm_1d",
            dropout=dropout,
        )
        self._generator = get_model(**model_kwargs)
        # print("generator: {:}".format(self._generator))

        # unknown token
        self.register_parameter(
            "_unknown_token",
            torch.nn.Parameter(torch.Tensor(1, time_embedding)),
        )

        # initialization
        trunc_normal_(
            [self._super_layer_embed, self._super_meta_embed, self._unknown_token],
            std=0.02,
        )

    @property
    def meta_timestamps(self):
        meta_timestamps = [self._meta_timestamps]
        for key in ("fixed", "learnt"):
            if self._append_meta_timestamps[key] is not None:
                meta_timestamps.append(self._append_meta_timestamps[key])
        return torch.cat(meta_timestamps)

    @property
    def super_meta_embed(self):
        meta_embed = [self._super_meta_embed]
        for key in ("fixed", "learnt"):
            if self._append_meta_embed[key] is not None:
                meta_embed.append(self._append_meta_embed[key])
        return torch.cat(meta_embed)

    def create_meta_embed(self):
        param = torch.Tensor(1, self._time_embed_dim)
        trunc_normal_(param, std=0.02)
        param = param.to(self._super_meta_embed.device)
        param = torch.nn.Parameter(param, True)
        return param

    def get_closest_meta_distance(self, timestamp):
        with torch.no_grad():
            distances = torch.abs(self.meta_timestamps - timestamp)
            return torch.min(distances).item()

    def replace_append_learnt(self, timestamp, meta_embed):
        self._append_meta_timestamps["learnt"] = timestamp
        self._append_meta_embed["learnt"] = meta_embed

    def append_fixed(self, timestamp, meta_embed):
        with torch.no_grad():
            device = self._super_meta_embed.device
            timestamp = timestamp.detach().clone().to(device)
            meta_embed = meta_embed.detach().clone().to(device)
            if self._append_meta_timestamps["fixed"] is None:
                self._append_meta_timestamps["fixed"] = timestamp
            else:
                self._append_meta_timestamps["fixed"] = torch.cat(
                    (self._append_meta_timestamps["fixed"], timestamp), dim=0
                )
            if self._append_meta_embed["fixed"] is None:
                self._append_meta_embed["fixed"] = meta_embed
            else:
                self._append_meta_embed["fixed"] = torch.cat(
                    (self._append_meta_embed["fixed"], meta_embed), dim=0
                )

    def forward_raw(self, timestamps):
        # timestamps is a batch of sequence of timestamps
        batch, seq = timestamps.shape
        timestamps = timestamps.unsqueeze(dim=-1)
        meta_timestamps = self.meta_timestamps.view(1, 1, -1)
        time_diffs = timestamps - meta_timestamps
        time_match_v, time_match_i = torch.min(torch.abs(time_diffs), dim=-1)
        # select corresponding meta-knowledge
        meta_match = torch.index_select(
            self.super_meta_embed, dim=0, index=time_match_i.view(-1)
        )
        meta_match = meta_match.view(batch, seq, -1)
        # create the probability
        time_probs = (1 / torch.exp(time_match_v * 10)).view(batch, seq, 1)
        if self.training:
            time_probs[:, -1, :] = 0
        unknown_token = self._unknown_token.view(1, 1, -1)
        raw_meta_embed = time_probs * meta_match + (1 - time_probs) * unknown_token

        meta_embed = self.meta_corrector(raw_meta_embed)
        # create joint embed
        num_layer, _ = self._super_layer_embed.shape
        meta_embed = meta_embed.view(batch, seq, 1, -1).expand(-1, -1, num_layer, -1)
        layer_embed = self._super_layer_embed.view(1, 1, num_layer, -1).expand(
            batch, seq, -1, -1
        )
        joint_embed = torch.cat((meta_embed, layer_embed), dim=-1)
        batch_weights = self._generator(joint_embed)
        batch_containers = []
        for seq_weights in torch.split(batch_weights, 1):
            seq_containers = []
            for weights in torch.split(seq_weights.squeeze(0), 1):
                weights = torch.split(weights.squeeze(0), 1)
                seq_containers.append(self._shape_container.translate(weights))
            batch_containers.append(seq_containers)
        return batch_containers

    def forward_candidate(self, input):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return "(_super_layer_embed): {:}, (_super_meta_embed): {:}, (_meta_timestamps): {:}".format(
            list(self._super_layer_embed.shape),
            list(self._super_meta_embed.shape),
            list(self._meta_timestamps.shape),
        )
