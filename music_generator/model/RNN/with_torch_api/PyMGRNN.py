import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from music_generator.model.RNN.with_torch_api import PyRNN

class PyMGRNN(nn.Module):
    def __init__(self, v_size: int, n_emb: int, h_dim: int, n_seq: int, bias: bool = False, device: str = "cpu"):
        super(PyMGRNN, self).__init__()
        self.bias = bias
        self.h_dim = h_dim
        self.v_size = v_size
        self.n_emb = n_emb
        self.n_seq = n_seq
        self.device = device

        self.emb = nn.Embedding(v_size, n_emb)
        self.rnn = PyRNN(in_dim=n_emb, out_dim=h_dim, n_unit=n_seq, bias=bias)
        self.fc = nn.Linear(h_dim, v_size)

    def forward(self, x: Tensor, y: Optional[Tensor] = None, h: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if h is None:
            h = torch.zeros(self.n_seq, x.size(0), self.h_dim, device=self.device)

        x = self.emb(x).to(self.device)
        x, h = self.rnn(x, h)

        if y is not None:
            logits = self.fc(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        else:
            logits = self.fc(x[:, [-1], :])
            loss = None

        return logits, loss
