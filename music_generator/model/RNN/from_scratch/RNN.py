import torch
import torch.nn as nn
from enum import Enum
from typing import Optional

class ACT_TYPES(Enum):
    TANH = "tanh"
    RELU = "relu"

class RNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, act: ACT_TYPES = ACT_TYPES.TANH, bias: bool = False, drop_rate: float = 0.0, device: str = "cpu"):
        super(RNN, self).__init__()
        self.act = act
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.bias = bias
        self.drop_rate = drop_rate

        self.i2h = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias, device=device)
        self.h2h = nn.Linear(in_features=out_dim, out_features=out_dim, bias=bias, device=device)
        self.dropout = nn.Dropout(drop_rate)

        self.act_fun = {
            ACT_TYPES.TANH: torch.tanh,
            ACT_TYPES.RELU: torch.relu
        }[act]

        self.init_parameters()
        self.to(device)

    def init_parameters(self) -> None:
        std = 1.0 / torch.sqrt(torch.tensor(self.in_dim, dtype=torch.float))
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor, xh: Optional[torch.Tensor] = None) -> torch.Tensor:
        if xh is None:
            xh = torch.zeros(x.size(0), self.out_dim, device=x.device)

        x = self.dropout(x) if self.drop_rate > 0 else x
        h = self.act_fun(self.i2h(x) + self.h2h(xh))

        return h