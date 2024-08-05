from music_generator.model.RNN.from_scratch import RNN
from music_generator.model.RNN.from_scratch.RNN import ACT_TYPES
import torch
import torch.nn as nn
from typing import List, Optional

class RNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_unit: int, act: ACT_TYPES = ACT_TYPES.TANH, bias: bool = False, drop_rate: float = 0.0, device: str = "cpu"):
        super(RNNBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_unit = n_unit
        self.device = device
        self.act = act
        self.bias = bias
        self.drop_rate = drop_rate

        self.units = nn.ModuleList([
            RNN(in_dim=in_dim, out_dim=out_dim, act=act, bias=bias, drop_rate=drop_rate, device=device) for _ in range(self.n_unit)
            ])

        self.to(device)

    def forward(self, x: torch.Tensor, xh: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape # x : [b, s, e]
        if xh is None:
            xh = torch.zeros(self.n_unit, b, self.out_dim, device=self.device) # [s, b, h] (0-> [b, h], 1->[b, h])

        xhs = [xh[i] for i in range(self.n_unit)]         # [x0, x1] --> [h0, h1]
        outs: List[torch.Tensor] = []                     # [h0, h1] --> [y0, y1]

        for idx in range(s):
            chr = x[:, idx, :]                            # [b, e]
            unit = self.units[idx]

            h = None if idx == 0 else xhs[idx - 1]        # [b, e] or [b, h]
            xhs[idx] = unit(chr, h)
            outs.append(xhs[idx])                         # [b, h] --> [b, 1, h]

        outs = torch.stack(outs, dim=1)                   # [b, 1, h] --> [b, s, h]
        return outs
