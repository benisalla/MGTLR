import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional, Tuple, List
from music_generator.model.LSTM.from_scratch import LSTM


class LSTMBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_unit: int, bias: bool = False, device: str = "cpu"):
        super(LSTMBlock, self).__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.n_unit: int = n_unit
        self.device: str = device

        self.units = nn.ModuleList([LSTM(in_dim=in_dim, out_dim=out_dim, bias=bias, device=device) for _ in range(n_unit)])


    # the following implementation is not working for the moment
    # i think the model graph is somehow closed somewhere (existance of loop)
    # for simplicity i have done a very simple implementation 
    
    # def forward(self,
    #             x: torch.Tensor, # [b, s, e]
    #             xh: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # []
    #             ) -> torch.Tensor:

    #     if xh is None:
    #         h0: torch.Tensor = Variable(torch.zeros(self.n_unit, x.size(0), self.out_dim, device=self.device))  # [s, b, o]
    #     else:
    #         h0: torch.Tensor = xh  # [s, b, o]

    #     outs: List[torch.Tensor] = []

    #     xhs: List[Tuple[torch.Tensor, torch.Tensor]] = [(h0[u, :, :], h0[u, :, :]) for u in range(self.n_unit)]

    #     for t in range(x.size(1)):
    #         for i, unit in enumerate(self.units):
    #             inp: torch.Tensor = x[:, t, :] if i == 0 else xhs[i - 1][0]
    #             uh: Tuple[torch.Tensor, torch.Tensor] = unit(inp, (xhs[i][0], xhs[i][1]))
    #             xhs[i] = uh

    #         outs.append(uh[0].unsqueeze(1))  # [b, h] ==> [b, 1, h]

    #     outs: torch.Tensor = torch.cat(outs, dim=1)  # [b, s, h]

    #     return outs
    
    
    def forward(self,
                x: torch.Tensor,  # [b, s, e]
                xh: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # []
                ) -> torch.Tensor:

        b, s, _ = x.shape   # [b, s, e]

        if xh is None:
            h0: torch.Tensor = Variable(torch.zeros(self.n_unit, b, self.out_dim, device=self.device))  # [s, b, o]
        else:
            h0: torch.Tensor = xh                                                                        # [s, b, o]
        
        xhs: List[Tuple[torch.Tensor, torch.Tensor]] = [(h0[u, :, :], h0[u, :, :]) for u in range(self.n_unit)]

        outs: List[torch.Tensor] = []  # [b, s, h]


        for i in range(s):
            chr = x[:, i, :]                            # [b, e]
            unit = self.units[i]                        # one LSTM unit
            
            xhs[i] = None if i == 0 else xhs[i - 1]
            uh = unit(chr, xhs[i])                      # [b, h], [b, h]
            xhs[i] = uh

            outs.append(uh[0])                          # [b, h] ==> [b, 1, h]

        outs: torch.Tensor = torch.stack(outs, dim=1)   # [b, s, h] 

        return outs