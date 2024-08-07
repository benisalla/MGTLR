import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class PyRNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_unit: int, bias: bool = False, act: str = "tanh"):
        super(PyRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=out_dim,
            num_layers=n_unit,
            nonlinearity=act,
            bias=bias,
            batch_first=True,
        )

    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        out, h = self.rnn(x, h)
        return out, h