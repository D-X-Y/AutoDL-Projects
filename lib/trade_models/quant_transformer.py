##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021 #
##################################################
from __future__ import division
from __future__ import print_function

import os, math, random
from collections import OrderedDict
import numpy as np
import pandas as pd
import copy
from functools import partial
from typing import Optional, Text
import logging

from qlib.utils import (
    unpack_archive_with_buffer,
    save_multiple_parts_file,
    get_or_create_path,
    drop_nan_by_y_index,
)
from qlib.log import get_module_logger, TimeInspector

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as th_data

import layers as xlayers
from log_utils import AverageMeter
from utils import count_parameters

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


default_net_config = dict(d_feat=6, embed_dim=48, depth=5, num_heads=4, mlp_ratio=4.0, qkv_bias=True, pos_drop=0.1)

default_opt_config = dict(
    epochs=200, lr=0.001, batch_size=2000, early_stop=20, loss="mse", optimizer="adam", num_workers=4
)


class QuantTransformer(Model):
    """Transformer-based Quant Model"""

    def __init__(self, net_config=None, opt_config=None, metric="", GPU=0, seed=None, **kwargs):
        # Set logger.
        self.logger = get_module_logger("QuantTransformer")
        self.logger.info("QuantTransformer PyTorch version...")

        # set hyper-parameters.
        self.net_config = net_config or default_net_config
        self.opt_config = opt_config or default_opt_config
        self.metric = metric
        self.device = torch.device("cuda:{:}".format(GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.logger.info(
            "Transformer parameters setting:"
            "\nnet_config : {:}"
            "\nopt_config : {:}"
            "\nmetric     : {:}"
            "\ndevice     : {:}"
            "\nseed       : {:}".format(
                self.net_config,
                self.opt_config,
                self.metric,
                self.device,
                self.seed,
            )
        )

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.use_gpu:
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

        self.model = TransformerModel(
            d_feat=self.net_config["d_feat"],
            embed_dim=self.net_config["embed_dim"],
            depth=self.net_config["depth"],
            pos_drop=self.net_config["pos_drop"],
        )
        self.logger.info("model: {:}".format(self.model))
        self.logger.info("model size: {:.3f} MB".format(count_parameters(self.model)))

        if self.opt_config["optimizer"] == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.opt_config["lr"])
        elif self.opt_config["optimizer"] == "adam":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.opt_config["lr"])
        else:
            raise NotImplementedError("optimizer {:} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.opt_config["loss"] == "mse":
            return F.mse_loss(pred[mask], label[mask])
        else:
            raise ValueError("unknown loss `{:}`".format(self.loss))

    def metric_fn(self, pred, label):
        # the metric score : higher is better
        if self.metric == "" or self.metric == "loss":
            return -self.loss_fn(pred, label)
        else:
            raise ValueError("unknown metric `{:}`".format(self.metric))

    def train_or_test_epoch(self, xloader, model, loss_fn, metric_fn, is_train, optimizer=None):
        if is_train:
            model.train()
        else:
            model.eval()
        score_meter, loss_meter = AverageMeter(), AverageMeter()
        for ibatch, (feats, labels) in enumerate(xloader):
            feats = feats.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # forward the network
            preds = model(feats)
            loss = loss_fn(preds, labels)
            with torch.no_grad():
                score = self.metric_fn(preds, labels)
                loss_meter.update(loss.item(), feats.size(0))
                score_meter.update(score.item(), feats.size(0))
            # optimize the network
            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
                optimizer.step()
        return loss_meter.avg, score_meter.avg

    def fit(
        self,
        dataset: DatasetH,
        save_path: Optional[Text] = None,
    ):
        def _prepare_dataset(df_data):
            return th_data.TensorDataset(
                torch.from_numpy(df_data["feature"].values).float(),
                torch.from_numpy(df_data["label"].values).squeeze().float(),
            )

        def _prepare_loader(dataset, shuffle):
            return th_data.DataLoader(
                dataset,
                batch_size=self.opt_config["batch_size"],
                drop_last=False,
                pin_memory=True,
                num_workers=self.opt_config["num_workers"],
                shuffle=shuffle,
            )

        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        train_dataset, valid_dataset, test_dataset = (
            _prepare_dataset(df_train),
            _prepare_dataset(df_valid),
            _prepare_dataset(df_test),
        )
        train_loader, valid_loader, test_loader = (
            _prepare_loader(train_dataset, True),
            _prepare_loader(valid_dataset, False),
            _prepare_loader(test_dataset, False),
        )

        save_path = get_or_create_path(save_path, return_dir=True)
        self.logger.info("Fit procedure for [{:}] with save path={:}".format(self.__class__.__name__, save_path))

        def _internal_test(ckp_epoch=None, results_dict=None):
            with torch.no_grad():
                train_loss, train_score = self.train_or_test_epoch(
                    train_loader, self.model, self.loss_fn, self.metric_fn, False, None
                )
                valid_loss, valid_score = self.train_or_test_epoch(
                    valid_loader, self.model, self.loss_fn, self.metric_fn, False, None
                )
                test_loss, test_score = self.train_or_test_epoch(
                    test_loader, self.model, self.loss_fn, self.metric_fn, False, None
                )
                xstr = "train-score={:.6f}, valid-score={:.6f}, test-score={:.6f}".format(
                    train_score, valid_score, test_score
                )
                if ckp_epoch is not None and isinstance(results_dict, dict):
                    results_dict["train"][ckp_epoch] = train_score
                    results_dict["valid"][ckp_epoch] = valid_score
                    results_dict["test"][ckp_epoch] = test_score
                return dict(train=train_score, valid=valid_score, test=test_score), xstr

        # Pre-fetch the potential checkpoints
        ckp_path = os.path.join(save_path, "{:}.pth".format(self.__class__.__name__))
        if os.path.exists(ckp_path):
            ckp_data = torch.load(ckp_path)
            import pdb

            pdb.set_trace()
        else:
            stop_steps, best_score, best_epoch = 0, -np.inf, -1
            start_epoch = 0
            results_dict = dict(train=OrderedDict(), valid=OrderedDict(), test=OrderedDict())
            _, eval_str = _internal_test(-1, results_dict)
            self.logger.info("Training from scratch, metrics@start: {:}".format(eval_str))

        for iepoch in range(start_epoch, self.opt_config["epochs"]):
            self.logger.info(
                "Epoch={:03d}/{:03d} ::==>> Best valid @{:03d} ({:.6f})".format(
                    iepoch, self.opt_config["epochs"], best_epoch, best_score
                )
            )

            train_loss, train_score = self.train_or_test_epoch(
                train_loader, self.model, self.loss_fn, self.metric_fn, True, self.train_optimizer
            )
            self.logger.info("Training :: loss={:.6f}, score={:.6f}".format(train_loss, train_score))

            current_eval_scores, eval_str = _internal_test(iepoch, results_dict)
            self.logger.info("Evaluating :: {:}".format(eval_str))

            if current_eval_scores["valid"] > best_score:
                stop_steps, best_epoch, best_score = 0, iepoch, current_eval_scores["valid"]
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.opt_config["early_stop"]:
                    self.logger.info("early stop at {:}-th epoch, where the best is @{:}".format(iepoch, best_epoch))
                    break
            save_info = dict(
                net_config=self.net_config,
                opt_config=self.opt_config,
                net_state_dict=self.model.state_dict(),
                opt_state_dict=self.train_optimizer.state_dict(),
                best_param=best_param,
                stop_steps=stop_steps,
                best_score=best_score,
                best_epoch=best_epoch,
                start_epoch=iepoch + 1,
            )
            torch.save(save_info, ckp_path)
        self.logger.info("The best score: {:.6f} @ {:02d}-th epoch".format(best_score, best_epoch))
        self.model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()
        self.fitted = True

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("The model is not fitted yet!")
        x_test = dataset.prepare("test", col_set="feature")
        index = x_test.index

        self.model.eval()
        x_values = x_test.values
        sample_num, batch_size = x_values.shape[0], self.opt_config["batch_size"]
        preds = []

        for begin in range(sample_num)[::batch_size]:

            if sample_num - begin < batch_size:
                end = sample_num
            else:
                end = begin + batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


# Real Model


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

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
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=mlp_drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = xlayers.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = xlayers.MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop)

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
        self.pos_embed = xlayers.PositionalEncoder(d_model=embed_dim, max_seq_len=max_seq_len, dropout=pos_drop)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
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

        cls_tokens = self.cls_token.expand(batch, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
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
