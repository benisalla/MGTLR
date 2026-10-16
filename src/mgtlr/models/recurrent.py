"""Vanilla RNN and LSTM language models, each in two flavours.

``impl="scratch"``: the cells are written by hand (two Linear layers and the
gate equations) and unrolled over time in Python. Slow, but every operation is
visible:

    RNN   h_t = act(W_ih x_t + W_hh h_{t-1})
    LSTM  i, f, g, o = split(W_ih x_t + W_hh h_{t-1})
          c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
          h_t = sigmoid(o) * tanh(c_t)

``impl="torch"``: the same maths through ``nn.RNN`` / ``nn.LSTM`` (fused
cuDNN kernels, much faster). The weight layouts match, so the unit tests copy
weights between the two and check the outputs are identical.

Both stack ``n_layers`` layers; each layer has **one** cell whose weights are
shared across all time steps.

    tokens [B, T] -> embedding [B, T, E] -> recurrent layers [B, T, H]
                  -> Linear -> logits [B, T, V]
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mgtlr.config import ModelConfig
from mgtlr.models.base import LanguageModel, ModelOutput, lm_loss

LSTMState = tuple[torch.Tensor, torch.Tensor]  # (h, c), each [n_layers, B, H]


def _init_uniform(module: nn.Module, hidden_dim: int) -> None:
    """Same initialisation as torch.nn.RNN / nn.LSTM: U(-1/sqrt(H), 1/sqrt(H))."""
    bound = 1.0 / math.sqrt(hidden_dim)
    for param in module.parameters():
        nn.init.uniform_(param, -bound, bound)


class RNNCell(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bias: bool, activation: str) -> None:
        super().__init__()
        self.i2h = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.act = torch.tanh if activation == "tanh" else torch.relu
        _init_uniform(self, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.act(self.i2h(x) + self.h2h(h))  # [B, H]


class LSTMCell(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bias: bool) -> None:
        super().__init__()
        self.i2h = nn.Linear(in_dim, 4 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
        _init_uniform(self, hidden_dim)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i, f, g, o = (self.i2h(x) + self.h2h(h)).chunk(4, dim=1)  # gate order of nn.LSTM
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ScratchRecurrent(nn.Module):
    """Stack of hand-written cells unrolled over time (batch_first, like nn.RNN)."""

    def __init__(
        self,
        kind: str,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        bias: bool,
        dropout: float,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.kind = kind
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        dims = [in_dim] + [hidden_dim] * (n_layers - 1)
        if kind == "lstm":
            self.cells = nn.ModuleList(LSTMCell(d, hidden_dim, bias) for d in dims)
        else:
            self.cells = nn.ModuleList(RNNCell(d, hidden_dim, bias, activation) for d in dims)
        self.dropout = nn.Dropout(dropout)  # between layers, like nn.RNN(dropout=...)

    def forward(self, x: torch.Tensor, state=None):
        B, T, _ = x.shape
        if state is None:
            zeros = x.new_zeros(self.n_layers, B, self.hidden_dim)
            state = (zeros, zeros.clone()) if self.kind == "lstm" else zeros
        new_h, new_c = [], []
        for layer, cell in enumerate(self.cells):
            if layer > 0:
                x = self.dropout(x)
            if self.kind == "lstm":
                h, c = state[0][layer], state[1][layer]
            else:
                h = state[layer]
            outputs = []
            for t in range(T):
                if self.kind == "lstm":
                    h, c = cell(x[:, t], (h, c))
                else:
                    h = cell(x[:, t], h)
                outputs.append(h)
            x = torch.stack(outputs, dim=1)  # [B, T, H]
            new_h.append(h)
            if self.kind == "lstm":
                new_c.append(c)
        h_n = torch.stack(new_h)
        return x, ((h_n, torch.stack(new_c)) if self.kind == "lstm" else h_n)


class RecurrentLM(LanguageModel):
    """Embedding -> recurrent layers -> Linear head. ``kind`` is "rnn" or "lstm"."""

    stateful = True
    kind = "rnn"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        c = config
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        layer_dropout = c.dropout if c.n_layers > 1 else 0.0
        if c.impl == "scratch":
            self.rnn = ScratchRecurrent(
                self.kind, c.d_model, c.hidden_dim, c.n_layers, c.bias, layer_dropout, c.activation
            )
        elif self.kind == "lstm":
            self.rnn = nn.LSTM(
                c.d_model, c.hidden_dim, c.n_layers, bias=c.bias, batch_first=True, dropout=layer_dropout
            )
        else:
            self.rnn = nn.RNN(
                c.d_model,
                c.hidden_dim,
                c.n_layers,
                nonlinearity=c.activation,
                bias=c.bias,
                batch_first=True,
                dropout=layer_dropout,
            )
        self.dropout = nn.Dropout(c.dropout)
        self.fc = nn.Linear(c.hidden_dim, c.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        state=None,
        label_smoothing: float = 0.0,
    ) -> ModelOutput:
        x, state = self.rnn(self.emb(idx), state)
        logits = self.fc(self.dropout(x))
        loss = lm_loss(logits, targets, label_smoothing) if targets is not None else None
        return ModelOutput(logits, loss, state)


class MGRNN(RecurrentLM):
    kind = "rnn"


class MGLSTM(RecurrentLM):
    kind = "lstm"
