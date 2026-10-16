"""All hyperparameters live here, grouped by what they control.

Configs are frozen dataclasses: they cannot be changed by accident after
creation, and they are saved inside every checkpoint so a run can be reproduced.

The three architectures do not share the same sweet spot, so each one has a
preset (``MODEL_PRESETS`` / ``TRAIN_PRESETS``) taken from the original project.
Command-line flags override individual values on top of the preset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

MODEL_TYPES = ("transformer", "lstm", "rnn")
RECURRENT_IMPLS = ("scratch", "torch")  # hand-written cells or torch.nn.RNN / nn.LSTM
ACTIVATIONS = ("tanh", "relu")  # vanilla RNN non-linearity
DEFAULT_DATA_DIR = Path("data")
DEFAULT_TOKENIZER_PATH = Path("tokenizer/mgt_tokenizer.model")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the ABC tunes come from and how the corpus is built."""

    # IrishMAN: 216k Irish folk tunes in ABC notation (MIT licence)
    dataset_id: str = "sander-wood/irishman"
    data_dir: Path = DEFAULT_DATA_DIR  # train.json / validation.json are downloaded here
    train_file: Path | None = None  # explicit JSON file, overrides data_dir/train.json
    val_file: Path | None = None
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH
    train_limit: int | None = None  # use only the first N tunes (quick runs)
    val_limit: int | None = None

    def __post_init__(self) -> None:
        for name in ("train_limit", "val_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape of the network. Fields that a model type does not use are ignored."""

    model_type: str  # "transformer" | "lstm" | "rnn"
    vocab_size: int
    d_model: int = 128  # token embedding size
    n_layers: int = 5  # transformer blocks or stacked recurrent layers
    dropout: float = 0.0
    bias: bool = False
    # transformer only
    head_dim: int = 32  # n_heads = d_model // head_dim
    ff_dim: int | None = None  # None = 4 * d_model
    max_seq_len: int = 1024  # size of the learned position table = context length
    # recurrent only
    hidden_dim: int = 256
    impl: str = "scratch"  # "scratch" (hand-written cells) | "torch" (cuDNN kernels)
    activation: str = "tanh"  # vanilla RNN only

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {self.model_type!r}")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if min(self.d_model, self.n_layers, self.head_dim, self.hidden_dim, self.max_seq_len) <= 0:
            raise ValueError("model sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.model_type == "transformer" and self.d_model % self.head_dim:
            raise ValueError("d_model must be divisible by head_dim")
        if self.impl not in RECURRENT_IMPLS:
            raise ValueError(f"impl must be one of {RECURRENT_IMPLS}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")

    @property
    def n_heads(self) -> int:
        return self.d_model // self.head_dim

    @property
    def ffn_dim(self) -> int:
        return self.ff_dim or 4 * self.d_model


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation and bookkeeping settings for one training run."""

    output_dir: Path = Path("checkpoints/run")
    seed: int = 42
    max_steps: int = 20_000  # optimizer steps
    batch_size: int = 32
    seq_len: int = 512  # tokens per training window
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps
    lr: float = 2e-4
    min_lr: float = 1e-6
    warmup_steps: int = 500
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    precision: str = "auto"  # "auto" | "fp32" | "bf16" | "fp16"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    log_every: int = 50  # optimizer steps between log lines
    eval_every: int = 500  # optimizer steps between validations
    eval_batches: int = 20  # batches averaged for one validation
    resume: Path | None = None  # checkpoint to continue from

    def __post_init__(self) -> None:
        positive = ("max_steps", "batch_size", "seq_len", "grad_accum_steps", "log_every")
        for name in (*positive, "eval_every", "eval_batches"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lr <= 0 or self.min_lr < 0 or self.min_lr > self.lr:
            raise ValueError("need 0 <= min_lr <= lr and lr > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16 or fp16")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")


# Defaults per architecture, taken from the original MGConfig / LSTMConfig.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "transformer": {"d_model": 128, "head_dim": 32, "n_layers": 5, "max_seq_len": 1024},
    "lstm": {"d_model": 128, "hidden_dim": 256, "n_layers": 2},
    "rnn": {"d_model": 128, "hidden_dim": 256, "n_layers": 2},
}
TRAIN_PRESETS: dict[str, dict[str, Any]] = {
    "transformer": {"seq_len": 512, "lr": 2e-4, "weight_decay": 1e-2, "beta2": 0.999},
    "lstm": {"seq_len": 200, "lr": 1e-3, "weight_decay": 1e-3, "beta2": 0.994},
    "rnn": {"seq_len": 200, "lr": 1e-3, "weight_decay": 1e-3, "beta2": 0.994},
}


def make_model_config(model_type: str, vocab_size: int, **overrides: Any) -> ModelConfig:
    """Preset for ``model_type`` with ``overrides`` (None values are ignored) on top."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
    values = {**MODEL_PRESETS[model_type], **_not_none(overrides)}
    return ModelConfig(model_type=model_type, vocab_size=vocab_size, **values)


def make_train_config(model_type: str, **overrides: Any) -> TrainConfig:
    """Preset for ``model_type`` with ``overrides`` (None values are ignored) on top."""
    return replace(TrainConfig(), **{**TRAIN_PRESETS[model_type], **_not_none(overrides)})


def config_to_dict(config: Any) -> dict[str, Any]:
    """Dataclass -> plain dict (Paths become strings) so it can go in a checkpoint."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}


def model_config_from_dict(values: dict[str, Any]) -> ModelConfig:
    known = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in values.items() if k in known})


def _not_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}
