# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import copy
from functools import partial
from sklearn.metrics import roc_auc_score, mean_squared_error
from typing import Optional
import logging

from qlib.utils import (
  unpack_archive_with_buffer,
  save_multiple_parts_file,
  create_save_path,
  drop_nan_by_y_index,
)
from qlib.log import get_module_logger, TimeInspector

import torch
import torch.nn as nn
import torch.optim as optim

from layers import DropPath, trunc_normal_

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class QuantTransformer(Model):
  """Transformer-based Quant Model

  """

  def __init__(
    self,
    d_feat=6,
    hidden_size=64,
    num_layers=2,
    dropout=0.0,
    n_epochs=200,
    lr=0.001,
    metric="",
    batch_size=2000,
    early_stop=20,
    loss="mse",
    optimizer="adam",
    GPU=0,
    seed=None,
    **kwargs
  ):
    # Set logger.
    self.logger = get_module_logger("QuantTransformer")
    self.logger.info("QuantTransformer pytorch version...")

    # set hyper-parameters.
    self.d_feat = d_feat
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.n_epochs = n_epochs
    self.lr = lr
    self.metric = metric
    self.batch_size = batch_size
    self.early_stop = early_stop
    self.optimizer = optimizer.lower()
    self.loss = loss
    self.device = torch.device("cuda:{:}".format(GPU) if torch.cuda.is_available() else "cpu")
    self.use_gpu = torch.cuda.is_available()
    self.seed = seed

    self.logger.info(
      "GRU parameters setting:"
      "\nd_feat : {}"
      "\nhidden_size : {}"
      "\nnum_layers : {}"
      "\ndropout : {}"
      "\nn_epochs : {}"
      "\nlr : {}"
      "\nmetric : {}"
      "\nbatch_size : {}"
      "\nearly_stop : {}"
      "\noptimizer : {}"
      "\nloss_type : {}"
      "\nvisible_GPU : {}"
      "\nuse_GPU : {}"
      "\nseed : {}".format(
        d_feat,
        hidden_size,
        num_layers,
        dropout,
        n_epochs,
        lr,
        metric,
        batch_size,
        early_stop,
        optimizer.lower(),
        loss,
        GPU,
        self.use_gpu,
        seed,
      )
    )

    if self.seed is not None:
      np.random.seed(self.seed)
      torch.manual_seed(self.seed)

    self.model = TransformerModel(d_feat=self.d_feat)
    if optimizer.lower() == "adam":
      self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    elif optimizer.lower() == "gd":
      self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
    else:
      raise NotImplementedError("optimizer {:} is not supported!".format(optimizer))

    self.fitted = False
    self.model.to(self.device)

  def mse(self, pred, label):
    loss = (pred - label) ** 2
    return torch.mean(loss)

  def loss_fn(self, pred, label):
    mask = ~torch.isnan(label)

    if self.loss == "mse":
      return self.mse(pred[mask], label[mask])

    raise ValueError("unknown loss `%s`" % self.loss)

  def metric_fn(self, pred, label):

    mask = torch.isfinite(label)

    if self.metric == "" or self.metric == "loss":
      return -self.loss_fn(pred[mask], label[mask])

    raise ValueError("unknown metric `%s`" % self.metric)

  def train_epoch(self, x_train, y_train):

    x_train_values = x_train.values
    y_train_values = np.squeeze(y_train.values)

    self.model.train()

    indices = np.arange(len(x_train_values))
    np.random.shuffle(indices)

    for i in range(len(indices))[:: self.batch_size]:

      if len(indices) - i < self.batch_size:
        break

      feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
      label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

      pred = self.model(feature)
      loss = self.loss_fn(pred, label)

      self.train_optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
      self.train_optimizer.step()

  def test_epoch(self, data_x, data_y):

    # prepare training data
    x_values = data_x.values
    y_values = np.squeeze(data_y.values)

    self.model.eval()

    scores = []
    losses = []

    indices = np.arange(len(x_values))
    import pdb; pdb.set_trace()

    for i in range(len(indices))[:: self.batch_size]:

      if len(indices) - i < self.batch_size:
        break

      feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
      label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

      pred = self.model(feature)
      loss = self.loss_fn(pred, label)
      losses.append(loss.item())

      score = self.metric_fn(pred, label)
      scores.append(score.item())

    return np.mean(losses), np.mean(scores)

  def fit(
    self,
    dataset: DatasetH,
    evals_result=dict(),
    verbose=True,
    save_path=None,
  ):

    df_train, df_valid, df_test = dataset.prepare(
      ["train", "valid", "test"],
      col_set=["feature", "label"],
      data_key=DataHandlerLP.DK_L,
    )

    x_train, y_train = df_train["feature"], df_train["label"]
    x_valid, y_valid = df_valid["feature"], df_valid["label"]

    if save_path == None:
      save_path = create_save_path(save_path)
    stop_steps = 0
    train_loss = 0
    best_score = -np.inf
    best_epoch = 0
    evals_result["train"] = []
    evals_result["valid"] = []

    # train
    self.logger.info("training...")
    self.fitted = True

    for step in range(self.n_epochs):
      self.logger.info("Epoch%d:", step)
      self.logger.info("training...")
      self.train_epoch(x_train, y_train)
      self.logger.info("evaluating...")
      train_loss, train_score = self.test_epoch(x_train, y_train)
      val_loss, val_score = self.test_epoch(x_valid, y_valid)
      self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
      evals_result["train"].append(train_score)
      evals_result["valid"].append(val_score)

      if val_score > best_score:
        best_score = val_score
        stop_steps = 0
        best_epoch = step
        best_param = copy.deepcopy(self.model.state_dict())
      else:
        stop_steps += 1
        if stop_steps >= self.early_stop:
          self.logger.info("early stop")
          break

    self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
    self.model.load_state_dict(best_param)
    torch.save(best_param, save_path)

    if self.use_gpu:
      torch.cuda.empty_cache()

  def predict(self, dataset):
    if not self.fitted:
      raise ValueError("model is not fitted yet!")

    x_test = dataset.prepare("test", col_set="feature")
    index = x_test.index
    self.model.eval()
    x_values = x_test.values
    sample_num = x_values.shape[0]
    preds = []

    for begin in range(sample_num)[:: self.batch_size]:

      if sample_num - begin < self.batch_size:
        end = sample_num
      else:
        end = begin + self.batch_size

      x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

      with torch.no_grad():
        if self.use_gpu:
          pred = self.model(x_batch).detach().cpu().numpy()
        else:
          pred = self.model(x_batch).detach().numpy()

      preds.append(pred)

    return pd.Series(np.concatenate(preds), index=index)


# Real Model


class Mlp(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop = nn.Dropout(drop)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


class Attention(nn.Module):
  def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads
    # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
    self.scale = qk_scale or head_dim ** -0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class Block(nn.Module):

  def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
         drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super().__init__()
    self.norm1 = norm_layer(dim)
    self.attn = Attention(
      dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
    # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

  def forward(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


class SimpleEmbed(nn.Module):

  def __init__(self, d_feat, embed_dim):
    super(SimpleEmbed, self).__init__()
    self.proj = nn.Linear(d_feat, embed_dim)

  def forward(self, x):
    import pdb; pdb.set_trace()
    B, C, H, W = x.shape
    # FIXME look at relaxing size constraints
    assert H == self.img_size[0] and W == self.img_size[1], \
      f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    x = self.proj(x).flatten(2).transpose(1, 2)
    return x


class TransformerModel(nn.Module):
  def __init__(self,
         d_feat: int,
         embed_dim: int = 64,
         depth: int = 4,
         num_heads: int = 4,
         mlp_ratio: float = 4.,
         qkv_bias: bool = True,
         qk_scale: Optional[float] = None,
         drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
    """
    Args:
      d_feat (int, tuple): input image size
      embed_dim (int): embedding dimension
      depth (int): depth of transformer
      num_heads (int): number of attention heads
      mlp_ratio (int): ratio of mlp hidden dim to embedding dim
      qkv_bias (bool): enable bias for qkv if True
      qk_scale (float): override default qk scale of head_dim ** -0.5 if set
      drop_rate (float): dropout rate
      attn_drop_rate (float): attention dropout rate
      drop_path_rate (float): stochastic depth rate
      norm_layer: (nn.Module): normalization layer
    """
    super(TransformerModel, self).__init__()
    self.embed_dim = embed_dim
    self.num_features = embed_dim
    norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

    self.input_embed = SimpleEmbed(d_feat, embed_dim=embed_dim)

    """
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    self.pos_drop = nn.Dropout(p=drop_rate)
    """

    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    self.blocks = nn.ModuleList([
      Block(
        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
      for i in range(depth)])
    self.norm = norm_layer(embed_dim)

    # regression head
    self.head = nn.Linear(self.num_features, 1)

    """
    trunc_normal_(self.pos_embed, std=.02)
    trunc_normal_(self.cls_token, std=.02)
    """
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  def forward_features(self, x):
    B = x.shape[0]
    x = self.input_embed(x)

    cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
      x = blk(x)

    x = self.norm(x)[:, 0]
    return x

  def forward(self, x):
    x = self.forward_features(x)
    x = self.head(x)
    return x
