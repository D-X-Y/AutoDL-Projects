import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
  # Attention Is All You Need: https://arxiv.org/pdf/1706.03762.pdf

  def __init__(self, d_model, max_seq_len):
    super(PositionalEncoder, self).__init__()
    self.d_model = d_model
    # create constant 'pe' matrix with values dependant on 
    # pos and i
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
      for i in range(0, d_model):
        div = 10000 ** ((i // 2) * 2 / d_model)
        value = pos / div
        if i % 2 == 0:
          pe[pos, i] = math.sin(value)
        else:
          pe[pos, i] = math.cos(value)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
 
  
  def forward(self, x):
    batch, seq, fdim = x.shape[:3]
    embeddings = self.pe[:, :seq, :fdim]
    return x + embeddings
