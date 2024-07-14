import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

class MGLayerNorm(nn.Module):
    def __init__(
            self, 
            n_dim: int, 
            bias: bool = True, 
            eps: float = 1e-5
        ):
        
        super(MGLayerNorm, self).__init__()
        self.eps: float = eps
        
        self.w: nn.Parameter = nn.Parameter(torch.ones(n_dim)) 
        self.b: Optional[nn.Parameter] = nn.Parameter(torch.zeros(n_dim)) if bias else None 

    def forward(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:
        
        return F.layer_norm(x, self.w.shape, self.w, self.b, 1e-5)