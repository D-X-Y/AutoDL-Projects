#
# This is used for the ablation studies:
# The meta-model in this file uses the traditional attention in
# transformer.
#
import torch

import torch.nn.functional as F

from xautodl.xlayers import super_core
from xautodl.xlayers import trunc_normal_
from xautodl.models.xcore import get_model


class MetaModel_TraditionalAtt(super_core.SuperModule):
    """Learning to Generate Models One Step Ahead (Meta Model Design)."""

    def __init__(
        self,
        shape_container,
        layer_dim,
        time_dim,
        meta_timestamps,
        dropout: float = 0.1,
        seq_length: int = None,
        interval: float = None,
        thresh: float = None,
    ):
        super(MetaModel_TraditionalAtt, self).__init__()
        self._shape_container = shape_container
        self._num_layers = len(shape_container)
        self._numel_per_layer = []
        for ilayer in range(self._num_layers):
            self._numel_per_layer.append(shape_container[ilayer].numel())
        self._raw_meta_timestamps = meta_timestamps
        assert interval is not None
        self._interval = interval
        self._thresh = interval * seq_length if thresh is None else thresh

        self.register_parameter(
            "_super_layer_embed",
            torch.nn.Parameter(torch.Tensor(self._num_layers, layer_dim)),
        )
        self.register_parameter(
            "_super_meta_embed",
            torch.nn.Parameter(torch.Tensor(len(meta_timestamps), time_dim)),
        )
        self.register_buffer("_meta_timestamps", torch.Tensor(meta_timestamps))
        self._time_embed_dim = time_dim
        self._append_meta_embed = dict(fixed=None, learnt=None)
        self._append_meta_timestamps = dict(fixed=None, learnt=None)

        self._tscalar_embed = super_core.SuperDynamicPositionE(
            time_dim, scale=1 / interval
        )

        # build transformer
        self._trans_att = super_core.SuperQKVAttention(
            in_q_dim=time_dim,
            in_k_dim=time_dim,
            in_v_dim=time_dim,
            num_heads=4,
            proj_dim=time_dim,
            qkv_bias=True,
            attn_drop=None,
            proj_drop=dropout,
        )

        model_kwargs = dict(
            config=dict(model_type="dual_norm_mlp"),
            input_dim=layer_dim + time_dim,
            output_dim=max(self._numel_per_layer),
            hidden_dims=[(layer_dim + time_dim) * 2] * 3,
            act_cls="gelu",
            norm_cls="layer_norm_1d",
            dropout=dropout,
        )
        self._generator = get_model(**model_kwargs)

        # initialization
        trunc_normal_(
            [self._super_layer_embed, self._super_meta_embed],
            std=0.02,
        )

    def get_parameters(self, time_embed, attention, generator):
        parameters = []
        if time_embed:
            parameters.append(self._super_meta_embed)
        if attention:
            parameters.extend(list(self._trans_att.parameters()))
        if generator:
            parameters.append(self._super_layer_embed)
            parameters.extend(list(self._generator.parameters()))
        return parameters

    @property
    def meta_timestamps(self):
        with torch.no_grad():
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

    @property
    def meta_length(self):
        return self.meta_timestamps.numel()

    def clear_fixed(self):
        self._append_meta_timestamps["fixed"] = None
        self._append_meta_embed["fixed"] = None

    def clear_learnt(self):
        self.replace_append_learnt(None, None)

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

    def gen_time_embed(self, timestamps):
        # timestamps is a batch of timestamps
        [B] = timestamps.shape
        # batch, seq = timestamps.shape
        timestamps = timestamps.view(-1, 1)
        meta_timestamps, meta_embeds = self.meta_timestamps, self.super_meta_embed
        timestamp_v_embed = meta_embeds.unsqueeze(dim=0)
        timestamp_q_embed = self._tscalar_embed(timestamps)
        timestamp_k_embed = self._tscalar_embed(meta_timestamps.view(1, -1))

        # create the mask
        mask = (
            torch.unsqueeze(timestamps, dim=-1) <= meta_timestamps.view(1, 1, -1)
        ) | (
            torch.abs(
                torch.unsqueeze(timestamps, dim=-1) - meta_timestamps.view(1, 1, -1)
            )
            > self._thresh
        )
        timestamp_embeds = self._trans_att(
            timestamp_q_embed, timestamp_k_embed, timestamp_v_embed, mask
        )
        return timestamp_embeds[:, -1, :]

    def gen_model(self, time_embeds):
        B, _ = time_embeds.shape
        # create joint embed
        num_layer, _ = self._super_layer_embed.shape
        # The shape of `joint_embed` is batch * num-layers * input-dim
        joint_embeds = torch.cat(
            (
                time_embeds.view(B, 1, -1).expand(-1, num_layer, -1),
                self._super_layer_embed.view(1, num_layer, -1).expand(B, -1, -1),
            ),
            dim=-1,
        )
        batch_weights = self._generator(joint_embeds)
        batch_containers = []
        for weights in torch.split(batch_weights, 1):
            batch_containers.append(
                self._shape_container.translate(torch.split(weights.squeeze(0), 1))
            )
        return batch_containers

    def forward_raw(self, timestamps, time_embeds, tembed_only=False):
        raise NotImplementedError

    def forward_candidate(self, input):
        raise NotImplementedError

    def easy_adapt(self, timestamp, time_embed):
        with torch.no_grad():
            timestamp = torch.Tensor([timestamp]).to(self._meta_timestamps.device)
            self.replace_append_learnt(None, None)
            self.append_fixed(timestamp, time_embed)

    def adapt(self, base_model, criterion, timestamp, x, y, lr, epochs, init_info):
        distance = self.get_closest_meta_distance(timestamp)
        if distance + self._interval * 1e-2 <= self._interval:
            return False, None
        x, y = x.to(self._meta_timestamps.device), y.to(self._meta_timestamps.device)
        with torch.set_grad_enabled(True):
            new_param = self.create_meta_embed()

            optimizer = torch.optim.Adam(
                [new_param], lr=lr, weight_decay=1e-5, amsgrad=True
            )
            timestamp = torch.Tensor([timestamp]).to(new_param.device)
            self.replace_append_learnt(timestamp, new_param)
            self.train()
            base_model.train()
            if init_info is not None:
                best_loss = init_info["loss"]
                new_param.data.copy_(init_info["param"].data)
            else:
                best_loss = 1e9
            with torch.no_grad():
                best_new_param = new_param.detach().clone()
            for iepoch in range(epochs):
                optimizer.zero_grad()
                time_embed = self.gen_time_embed(timestamp.view(1))
                match_loss = F.l1_loss(new_param, time_embed)

                [container] = self.gen_model(new_param.view(1, -1))
                y_hat = base_model.forward_with_container(x, container)
                meta_loss = criterion(y_hat, y)
                loss = meta_loss + match_loss
                loss.backward()
                optimizer.step()
                if meta_loss.item() < best_loss:
                    with torch.no_grad():
                        best_loss = meta_loss.item()
                        best_new_param = new_param.detach().clone()
        self.easy_adapt(timestamp, best_new_param)
        return True, best_loss

    def extra_repr(self) -> str:
        return "(_super_layer_embed): {:}, (_super_meta_embed): {:}, (_meta_timestamps): {:}".format(
            list(self._super_layer_embed.shape),
            list(self._super_meta_embed.shape),
            list(self._meta_timestamps.shape),
        )
