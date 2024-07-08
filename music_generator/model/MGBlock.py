import torch.nn as nn
from music_generator.model.MGFeedForward import MGFeedForward
from music_generator.model.MGLayerNorm import MGLayerNorm
from music_generator.model.MGMaskAttention import MGMaskAttention

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