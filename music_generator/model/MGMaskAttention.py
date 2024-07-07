import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MGMaskAttention(nn.Module):
    def __init__(self, n_embd, n_head, b_size, drop_rate, bias):
        super().__init__()
        assert n_embd % n_head == 0 ; "n_embd % n_head should be 0."

        self.qkv_ln = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj_ln = nn.Linear(n_embd, n_embd, bias=bias)

        self.atten_drop = nn.Dropout(drop_rate)
        self.res_drop = nn.Dropout(drop_rate)
        self.n_head = n_head
        self.n_embd = n_embd
        self.drop_rate = drop_rate
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(b_size, b_size))
                                        .view(1, 1, b_size, b_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.qkv_ln(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.drop_rate if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.atten_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.res_drop(self.proj_ln(y))
        return y
