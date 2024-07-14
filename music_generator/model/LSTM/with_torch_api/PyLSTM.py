import torch
from torch import nn
from typing import Optional, Tuple

class PyLSTM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_unit: int, bias: bool = False) -> None:
        super(PyLSTM, self).__init__()
        self.n_unit: int = n_unit
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=out_dim,
            num_layers=n_unit,
            bias=bias,
            batch_first=True
        )

    def forward(self, 
                x: torch.Tensor, 
                h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if h is None:
            batch_size = x.size(0)
            h = (torch.zeros(self.n_unit, batch_size, self.out_dim, device=x.device),
                torch.zeros(self.n_unit, batch_size, self.out_dim, device=x.device))

        y: torch.Tensor
        h: Tuple[torch.Tensor, torch.Tensor]
        y, h = self.lstm(x, h)
        return y, h