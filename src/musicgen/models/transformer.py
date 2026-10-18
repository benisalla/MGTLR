"""GPT-style decoder-only Transformer, written from scratch (the original MGTransformer).

    tokens [B, T] -> token + position embeddings [B, T, C]
                  -> n_layers x (LayerNorm -> causal self-attention -> +residual,
                                 LayerNorm -> MLP -> +residual)
                  -> LayerNorm -> lm_head (tied with the token embedding) -> logits [B, T, V]

Parameter names are the same as in the original code, so checkpoints trained
with the legacy scripts load into this class (see ``musicgen.checkpoint``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from musicgen.config import ModelConfig
from musicgen.models.base import LanguageModel, ModelOutput, lm_loss


class MGLayerNorm(nn.Module):
    """LayerNorm with an optional bias (torch's LayerNorm always has one)."""

    def __init__(self, n_dim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(n_dim))
        self.b = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.w.shape, self.w, self.b, self.eps)


class MGMaskAttention(nn.Module):
    """Multi-head causal self-attention (fused scaled_dot_product_attention kernel)."""

    def __init__(self, n_embd: int, n_head: int, drop_rate: float, bias: bool) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.n_embd = n_embd
        self.drop_rate = drop_rate
        self.qkv_ln = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj_ln = nn.Linear(n_embd, n_embd, bias=bias)
        self.res_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_ln(x).split(self.n_embd, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_rate if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.res_drop(self.proj_ln(y))


class MGFeedForward(nn.Module):
    def __init__(self, n_embd: int, ff_dim: int, drop_rate: float, bias: bool) -> None:
        super().__init__()
        self.fc_ln = nn.Linear(n_embd, ff_dim, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(ff_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc_ln(x))))


class MGBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, n_embd: int, n_head: int, ff_dim: int, drop_rate: float, bias: bool) -> None:
        super().__init__()
        self.ln1 = MGLayerNorm(n_embd, bias=bias)
        self.attn = MGMaskAttention(n_embd, n_head, drop_rate, bias)
        self.ln2 = MGLayerNorm(n_embd, bias=bias)
        self.ff = MGFeedForward(n_embd, ff_dim, drop_rate, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class MGTransformer(LanguageModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        c = config
        self.decoder = nn.ModuleDict(
            {
                "tok_emb": nn.Embedding(c.vocab_size, c.d_model),
                "pos_emb": nn.Embedding(c.max_seq_len, c.d_model),
                "dropout": nn.Dropout(c.dropout),
                "blocks": nn.ModuleList(
                    MGBlock(c.d_model, c.n_heads, c.ffn_dim, c.dropout, c.bias) for _ in range(c.n_layers)
                ),
                "f_ln": MGLayerNorm(c.d_model, bias=c.bias),
            }
        )
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.decoder.tok_emb.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2: scale the residual projections by 1/sqrt(2 * n_layers)
        for name, param in self.named_parameters():
            if name.endswith(("proj.weight", "proj_ln.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        state: None = None,
        label_smoothing: float = 0.0,
    ) -> ModelOutput:
        T = idx.size(1)
        if T > self.config.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len={self.config.max_seq_len}")
        pos = torch.arange(T, device=idx.device)
        x = self.decoder.dropout(self.decoder.tok_emb(idx) + self.decoder.pos_emb(pos))
        for block in self.decoder.blocks:
            x = block(x)
        logits = self.lm_head(self.decoder.f_ln(x))
        loss = lm_loss(logits, targets, label_smoothing) if targets is not None else None
        return ModelOutput(logits, loss, None)
