import torch.nn as nn
from music_generator.model.TRF.MGFeedForward import MGFeedForward
from music_generator.model.TRF.MGLayerNorm import MGLayerNorm
from music_generator.model.TRF.MGMaskAttention import MGMaskAttention

class MGBlock(nn.Module):
    def __init__(self, n_embd, n_head, b_size, ff_dim, drop_rate, bias):
        super().__init__()
        self.ln1 = MGLayerNorm(n_dim=n_embd, bias=bias)
        self.attn = MGMaskAttention(n_embd=n_embd, n_head=n_head, b_size=b_size, drop_rate=drop_rate, bias=bias)
        self.ln2 = MGLayerNorm(n_dim=n_embd, bias=bias)
        self.ff = MGFeedForward(n_embd=n_embd, ff_dim=ff_dim, drop_rate=drop_rate, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# import torch
# from torch import nn
# from music_generator.model.TRF.MGFeedForward import MGFeedForward
# from music_generator.model.TRF.MGLayerNorm import MGLayerNorm
# from music_generator.model.TRF.MGMaskAttention import MGMaskAttention

# class MGBlock(nn.Module):
#     def __init__(
#             self, 
#             n_embd: int, 
#             n_head: int, 
#             b_size: int, 
#             ff_dim: int, 
#             drop_rate: float, 
#             bias: bool
#         ):
        
#         super(MGBlock, self).__init__()
#         self.ln1: MGLayerNorm = MGLayerNorm(n_dim=n_embd, bias=bias)
#         self.attn: MGMaskAttention = MGMaskAttention(n_embd=n_embd, n_head=n_head, b_size=b_size, drop_rate=drop_rate, bias=bias)
#         self.ln2: MGLayerNorm = MGLayerNorm(n_dim=n_embd, bias=bias)
#         self.ff: MGFeedForward = MGFeedForward(n_embd=n_embd, ff_dim=ff_dim, drop_rate=drop_rate, bias=bias)

#     def forward(
#             self, 
#             x: torch.Tensor
#         ) -> torch.Tensor:
        
#         x = x + self.attn(self.ln1(x))
#         x = x + self.ff(self.ln2(x))
#         return x
