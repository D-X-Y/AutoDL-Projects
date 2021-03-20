#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
from __future__ import division
from __future__ import print_function

import math
from functools import partial
from typing import Optional, Text

import torch
import torch.nn as nn
import torch.nn.functional as F

import xlayers


DEFAULT_NET_CONFIG = dict(
    d_feat=6,
    embed_dim=64,
    depth=5,
    num_heads=4,
    mlp_ratio=4.0,
    qkv_bias=True,
    pos_drop=0.0,
    mlp_drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
)


# Real Model


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or math.sqrt(head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        mlp_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=mlp_drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            xlayers.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = xlayers.MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SimpleEmbed(nn.Module):
    def __init__(self, d_feat, embed_dim):
        super(SimpleEmbed, self).__init__()
        self.d_feat = d_feat
        self.embed_dim = embed_dim
        self.proj = nn.Linear(d_feat, embed_dim)

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F*T] -> [N, F, T]
        x = x.permute(0, 2, 1)  # [N, F, T] -> [N, T, F]
        out = self.proj(x) * math.sqrt(self.embed_dim)
        return out


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_feat: int = 6,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        pos_drop: float = 0.0,
        mlp_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Optional[nn.Module] = None,
        max_seq_len: int = 65,
    ):
        """
        Args:
          d_feat (int, tuple): input image size
          embed_dim (int): embedding dimension
          depth (int): depth of transformer
          num_heads (int): number of attention heads
          mlp_ratio (int): ratio of mlp hidden dim to embedding dim
          qkv_bias (bool): enable bias for qkv if True
          qk_scale (float): override default qk scale of head_dim ** -0.5 if set
          pos_drop (float): dropout rate for the positional embedding
          mlp_drop_rate (float): the dropout rate for MLP layers in a block
          attn_drop_rate (float): attention dropout rate
          drop_path_rate (float): stochastic depth rate
          norm_layer: (nn.Module): normalization layer
        """
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.input_embed = SimpleEmbed(d_feat, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = xlayers.PositionalEncoder(
            d_model=embed_dim, max_seq_len=max_seq_len, dropout=pos_drop
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop_rate,
                    mlp_drop=mlp_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # regression head
        self.head = nn.Linear(self.num_features, 1)

        xlayers.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xlayers.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        batch, flatten_size = x.shape
        feats = self.input_embed(x)  # batch * 60 * 64

        cls_tokens = self.cls_token.expand(
            batch, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        feats_w_ct = torch.cat((cls_tokens, feats), dim=1)
        feats_w_tp = self.pos_embed(feats_w_ct)

        xfeats = feats_w_tp
        for block in self.blocks:
            xfeats = block(xfeats)

        xfeats = self.norm(xfeats)[:, 0]
        return xfeats

    def forward(self, x):
        feats = self.forward_features(x)
        predicts = self.head(feats).squeeze(-1)
        return predicts


def get_transformer(config):
    if not isinstance(config, dict):
        raise ValueError("Invalid Configuration: {:}".format(config))
    name = config.get("name", "basic")
    if name == "basic":
        model = TransformerModel(
            d_feat=config.get("d_feat"),
            embed_dim=config.get("embed_dim"),
            depth=config.get("depth"),
            num_heads=config.get("num_heads"),
            mlp_ratio=config.get("mlp_ratio"),
            qkv_bias=config.get("qkv_bias"),
            qk_scale=config.get("qkv_scale"),
            pos_drop=config.get("pos_drop"),
            mlp_drop_rate=config.get("mlp_drop_rate"),
            attn_drop_rate=config.get("attn_drop_rate"),
            drop_path_rate=config.get("drop_path_rate"),
            norm_layer=config.get("norm_layer", None),
        )
    else:
        raise ValueError("Unknown model name: {:}".format(name))
    return model
