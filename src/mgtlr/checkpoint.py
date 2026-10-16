"""Save and load checkpoints.

A checkpoint is self-contained: model weights, model config, the tokenizer
(merges + special tokens) and, for resuming, optimizer state and step count.

Checkpoints written by the original project (``{"model", "model_args", ...}``,
Transformer only, tokenizer in a separate ``.model`` file) load too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from mgtlr.config import ModelConfig, config_to_dict, model_config_from_dict
from mgtlr.models import LanguageModel, build_model
from mgtlr.tokenizer import MGTokenizer

FORMAT_VERSION = 1


@dataclass
class LoadedCheckpoint:
    model: LanguageModel
    tokenizer: MGTokenizer
    raw: dict[str, Any] = field(repr=False)

    @property
    def step(self) -> int:
        return int(self.raw.get("step", 0))


def save_checkpoint(
    path: str | Path,
    model: LanguageModel,
    tokenizer: MGTokenizer,
    optimizer: torch.optim.Optimizer | None = None,
    **extra: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_config": config_to_dict(model.config),
        "model": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        **extra,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint
    return path


def legacy_model_config(model_args: dict[str, Any]) -> ModelConfig:
    """Map the original MGTransformer.get_init_args() to a ModelConfig."""
    return ModelConfig(
        model_type="transformer",
        vocab_size=model_args["v_size"],
        d_model=model_args["n_embd"],
        head_dim=model_args["h_dim"],
        n_layers=model_args["n_block"],
        max_seq_len=model_args["max_seq_len"],
        dropout=model_args.get("drop_rate", 0.0),
        bias=model_args.get("bias", False),
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    tokenizer_path: str | Path | None = None,
) -> LoadedCheckpoint:
    """Rebuild model + tokenizer. ``tokenizer_path`` is only needed for legacy files."""
    raw = torch.load(path, map_location=device, weights_only=False)
    if "model_config" in raw:
        config = model_config_from_dict(raw["model_config"])
        tokenizer = MGTokenizer.from_dict(raw["tokenizer"])
    elif "model_args" in raw:  # original project format
        config = legacy_model_config(raw["model_args"])
        if tokenizer_path is None:
            raise ValueError("legacy checkpoints need tokenizer_path (the .model file)")
        tokenizer = MGTokenizer.load(tokenizer_path)
    else:
        raise ValueError(f"{path} is not an MGTLR checkpoint")
    if tokenizer_path is not None and "model_config" in raw:
        tokenizer = MGTokenizer.load(tokenizer_path)
    model = build_model(config)
    model.load_state_dict(raw["model"])
    model.to(device).eval()
    return LoadedCheckpoint(model=model, tokenizer=tokenizer, raw=raw)
