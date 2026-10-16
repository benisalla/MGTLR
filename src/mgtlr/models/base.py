"""What the three models share: the forward contract, the loss and sampling.

Every model maps token ids ``[B, T]`` to next-token logits ``[B, T, V]``.
Recurrent models also carry a hidden ``state`` between calls, so generation
feeds them one new token at a time; the Transformer has no state and re-reads
its (cropped) context at every step.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mgtlr.config import ModelConfig

IGNORE_INDEX = -1  # target value excluded from the loss (same as the original code)


class ModelOutput(NamedTuple):
    logits: torch.Tensor  # [B, T, V]
    loss: torch.Tensor | None  # mean cross-entropy over targets != IGNORE_INDEX
    state: Any = None  # recurrent hidden state, None for the Transformer


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        label_smoothing=label_smoothing,
    )


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token per row from ``[B, V]`` logits. ``temperature=0`` means greedy."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


class LanguageModel(nn.Module):
    """Base class: subclasses implement ``forward(idx, targets=None, state=None)``."""

    stateful: bool = False  # True for RNN / LSTM

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Extend ``idx`` ``[B, T]`` by up to ``max_new_tokens`` sampled tokens.

        Stops early once every row has produced ``eos_id`` (rows that finished
        first are padded with ``eos_id``).
        """
        was_training = self.training
        self.eval()
        out = idx
        state = None
        step_input = idx
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            if self.stateful:
                logits, _, state = self(step_input, state=state)
            else:
                logits, _, _ = self(out[:, -self.config.max_seq_len :])
            next_token = sample_next(logits[:, -1, :], temperature, top_k, generator)
            if eos_id is not None:
                next_token = next_token.masked_fill(finished[:, None], eos_id)
                finished |= next_token.squeeze(1) == eos_id
            out = torch.cat([out, next_token], dim=1)
            step_input = next_token
            if eos_id is not None and bool(finished.all()):
                break
        self.train(was_training)
        return out
