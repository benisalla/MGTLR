import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional, Tuple


class LSTM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False, device: str = 'cpu'):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.device = device

        self.i2h = nn.Linear(in_dim, out_dim * 4, bias=bias).to(device)   # [b, s, 4*o]
        self.h2h = nn.Linear(out_dim, out_dim * 4, bias=bias).to(device)  # [b, o, 4*o]

        self.init_parameters()

    def init_parameters(self) -> None:
        std = 1.0 / torch.sqrt(torch.tensor(self.in_dim, dtype=torch.float))
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, x: torch.Tensor, xh: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if xh is None:
            xh = Variable(x.new_zeros(x.size(0), self.out_dim))  # [b, o]
            xh = (xh, xh)

        xh, xc = xh

        gts = self.i2h(x) + self.h2h(xh)                # [b, s, 4*o]
        igt, fgt, cgt, ogt = gts.chunk(4, 1)            # gts ==> [i, f, c, o]

        i = torch.sigmoid(igt)  # [b, s, o]
        f = torch.sigmoid(fgt)  # [b, s, o]
        g = torch.tanh(cgt)     # [b, s, o]
        o = torch.sigmoid(ogt)  # [b, s, o]

        yc = xc * f + i * g       # [b, s, o]
        yh = o * torch.tanh(yc)   # [b, s, o]

        return (yh, yc)