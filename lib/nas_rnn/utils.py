import torch
import torch.nn as nn
import os, shutil
import numpy as np


def repackage_hidden(h):
  if isinstance(h, torch.Tensor):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, use_cuda):
  nbatch = data.size(0) // bsz
  data = data.narrow(0, 0, nbatch * bsz)
  data = data.view(bsz, -1).t().contiguous()
  if use_cuda: return data.cuda()
  else     : return data


def get_batch(source, i, seq_len):
  seq_len = min(seq_len, len(source) - 1 - i)
  data    = source[i:i+seq_len].clone()
  target  = source[i+1:i+1+seq_len].clone()
  return data, target



def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    mask.requires_grad_(True)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
    padding_idx = -1
  X = torch.nn.functional.embedding(
        words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse)
  return X


class LockedDropout(nn.Module):
  def __init__(self):
    super(LockedDropout, self).__init__()

  def forward(self, x, dropout=0.5):
    if not self.training or not dropout:
      return x
    m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
    mask = m.div_(1 - dropout).detach()
    mask = mask.expand_as(x)
    return mask * x


def mask2d(B, D, keep_prob, cuda=True):
  m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
  if cuda: return m.cuda()
  else   : return m
