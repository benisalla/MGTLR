import torch
import torch.nn as nn
import torch.nn.functional as F

class MGLayerNorm(nn.Module):
    def __init__(self, n_dim, bias=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(n_dim))
        self.b = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.w.shape, self.w, self.b, 1e-5)