#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.06 #
#####################################################
# Vision Transformer: arxiv.org/pdf/2010.11929.pdf  #
#####################################################
import copy, math
from functools import partial
from typing import Optional, Text, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from xautodl import spaces
from xautodl import xlayers
from xautodl.xlayers import weight_init


class SuperQuaT(xlayers.SuperModule):
    """The super transformer for transformer."""

    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_multiplier=4,
        channels=3,
        dropout=0.0,
        att_dropout=0.0,
    ):
        super(SuperQuaT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = xlayers.SuperSequential(
            xlayers.SuperReArrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            xlayers.SuperLinear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # build the transformer encode layers
        layers = []
        for ilayer in range(depth):
            layers.append(
                xlayers.SuperTransformerEncoderLayer(
                    dim,
                    heads,
                    False,
                    mlp_multiplier,
                    dropout=dropout,
                    att_dropout=att_dropout,
                )
            )
        self.backbone = xlayers.SuperSequential(*layers)
        self.cls_head = xlayers.SuperSequential(
            xlayers.SuperLayerNorm1D(dim), xlayers.SuperLinear(dim, num_classes)
        )

        weight_init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    @property
    def abstract_search_space(self):
        raise NotImplementedError

    def apply_candidate(self, abstract_child: spaces.VirtualNode):
        super(SuperQuaT, self).apply_candidate(abstract_child)
        raise NotImplementedError

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        tensors = self.to_patch_embedding(input)
        batch, seq, _ = tensors.shape

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        feats = torch.cat((cls_tokens, tensors), dim=1)
        feats = feats + self.pos_embedding[:, : seq + 1, :]
        feats = self.dropout(feats)

        feats = self.backbone(feats)

        x = feats[:, 0]  # the features for cls-token

        return self.cls_head(x)


def get_transformer(config):
    if isinstance(config, str) and config.lower() in name2config:
        config = name2config[config.lower()]
    if not isinstance(config, dict):
        raise ValueError("Invalid Configuration: {:}".format(config))
    model_type = config.get("type", "vit").lower()
    if model_type == "vit":
        model = SuperQuaT(
            image_size=config.get("image_size"),
            patch_size=config.get("patch_size"),
            num_classes=config.get("num_classes"),
            dim=config.get("dim"),
            depth=config.get("depth"),
            heads=config.get("heads"),
            dropout=config.get("dropout"),
            att_dropout=config.get("att_dropout"),
        )
    else:
        raise ValueError("Unknown model type: {:}".format(model_type))
    return model
