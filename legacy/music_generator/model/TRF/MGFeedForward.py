import torch
from torch import nn

class MGFeedForward(nn.Module):
    def __init__(
            self, 
            n_embd: int, 
            ff_dim: int, 
            drop_rate: float, 
            bias: bool
        ):
        
        super(MGFeedForward, self).__init__()
        self.fc_ln: nn.Linear = nn.Linear(n_embd, ff_dim, bias=bias)    # [b, *, n_embd] -> [b, *, ff_dim]
        self.act: nn.GELU = nn.GELU()
        self.proj: nn.Linear = nn.Linear(ff_dim, n_embd, bias=bias)     # [b, *, ff_dim] -> [b, *, n_embd]
        self.dropout: nn.Dropout = nn.Dropout(drop_rate)

    def forward(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:
        
        x: torch.Tensor = self.act(self.fc_ln(x))       # [b, *, ff_dim]
        x: torch.Tensor = self.dropout(self.proj(x))    # [b, *, n_embd]
        
        return x
