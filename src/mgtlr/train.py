"""Train a Transformer, LSTM or RNN on ABC tunes. One loop for all three.

The file reads top to bottom:

1. setup helpers  - seeding, device, mixed precision
2. optimisation   - AdamW (no decay on norms/biases) + warmup/cosine schedule
3. evaluation     - loss, next-token accuracy, perplexity on fixed val windows
4. train()        - the loop: takes ready-made objects, easy to test
5. CLI            - main() builds everything from command-line flags

Run it with:  mgtlr-train --model transformer      (or lstm / rnn; see --help)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from mgtlr.checkpoint import save_checkpoint
from mgtlr.config import (
    ACTIVATIONS,
    DEFAULT_DATA_DIR,
    DEFAULT_TOKENIZER_PATH,
    MODEL_TYPES,
    RECURRENT_IMPLS,
    DataConfig,
    TrainConfig,
    config_to_dict,
    make_model_config,
    make_train_config,
)
from mgtlr.data import TokenStream, build_streams, format_corpus, load_tunes, resolve_split_file
from mgtlr.models import LanguageModel, build_model
from mgtlr.tokenizer import MGTokenizer

logger = logging.getLogger(__name__)

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"
HISTORY_FILE = "history.json"


# ---------------------------------------------------------------------------
# 1. Setup helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(name: str, device: torch.device) -> str:
    """'auto' = bf16 on GPUs that support it, else fp16 on CUDA, else fp32."""
    if name != "auto":
        if name in {"bf16", "fp16"} and device.type != "cuda":
            raise ValueError(f"{name} needs a CUDA device")
        return name
    if device.type == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def autocast(device: torch.device, precision: str) -> AbstractContextManager:
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


# ---------------------------------------------------------------------------
# 2. Optimisation
# ---------------------------------------------------------------------------


def make_optimizer(model: LanguageModel, cfg: TrainConfig) -> AdamW:
    """Weight decay on matrices/embeddings only; norms and biases are not decayed."""
    decay, no_decay = [], []
    for param in model.parameters():  # tied weights appear once
        if param.requires_grad:
            (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to ``lr``, then cosine decay to ``min_lr`` at ``max_steps``."""
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: LanguageModel,
    stream: TokenStream,
    cfg: TrainConfig,
    device: torch.device,
    precision: str = "fp32",
) -> dict[str, float]:
    """Average over ``eval_batches`` windows; the same windows every call (fixed seed)."""
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(cfg.seed + 1)
    total_loss = total_correct = total_tokens = 0.0
    for _ in range(cfg.eval_batches):
        x, y = stream.sample_batch(cfg.batch_size, cfg.seq_len, generator)
        x, y = x.to(device), y.to(device)
        with autocast(device, precision):
            out = model(x, y)
        total_loss += out.loss.item() * y.numel()
        total_correct += (out.logits.argmax(-1) == y).sum().item()
        total_tokens += y.numel()
    model.train(was_training)
    loss = total_loss / total_tokens
    return {"loss": loss, "accuracy": total_correct / total_tokens, "perplexity": math.exp(min(loss, 50))}


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = float("inf")
    history: list[dict[str, Any]] = field(default_factory=list)


def train(
    model: LanguageModel,
    tokenizer: MGTokenizer,
    train_stream: TokenStream,
    val_stream: TokenStream,
    cfg: TrainConfig,
    extra_metadata: dict[str, Any] | None = None,
) -> TrainState:
    """Run ``cfg.max_steps`` optimizer steps; write last.pt / best.pt / history.json."""
    if model.config.model_type == "transformer" and cfg.seq_len > model.config.max_seq_len:
        raise ValueError(f"seq_len={cfg.seq_len} > max_seq_len={model.config.max_seq_len}")
    device = resolve_device(cfg.device)
    precision = resolve_precision(cfg.precision, device)
    model.to(device).train()
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = TrainState()
    if cfg.resume is not None:
        raw = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(raw["model"])
        if "optimizer" in raw:
            optimizer.load_state_dict(raw["optimizer"])
        state = TrainState(raw.get("step", 0), raw.get("best_val_loss", float("inf")), raw.get("history", []))
        logger.info("resumed from %s at step %d", cfg.resume, state.step)

    metadata = {"train_config": config_to_dict(cfg), **(extra_metadata or {})}

    def checkpoint(name: str) -> None:
        save_checkpoint(
            out_dir / name,
            model,
            tokenizer,
            optimizer,
            step=state.step,
            best_val_loss=state.best_val_loss,
            history=state.history,
            **metadata,
        )

    logger.info(
        "training %s (%s params) on %s, precision %s, %d train tokens",
        model.config.model_type,
        f"{model.num_parameters():,}",
        device,
        precision,
        len(train_stream),
    )
    generator = torch.Generator().manual_seed(cfg.seed + state.step)
    window_loss = window_correct = window_tokens = 0.0
    start = time.perf_counter()

    while state.step < cfg.max_steps:
        lr = lr_at(state.step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(cfg.grad_accum_steps):
            x, y = train_stream.sample_batch(cfg.batch_size, cfg.seq_len, generator)
            x, y = x.to(device), y.to(device)
            with autocast(device, precision):
                out = model(x, y, label_smoothing=cfg.label_smoothing)
            scaler.scale(out.loss / cfg.grad_accum_steps).backward()
            window_loss += out.loss.item() * y.numel()
            window_correct += (out.logits.detach().argmax(-1) == y).sum().item()
            window_tokens += y.numel()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        state.step += 1

        if state.step % cfg.log_every == 0 or state.step == cfg.max_steps:
            elapsed = time.perf_counter() - start
            logger.info(
                "step %d/%d | loss %.4f | acc %.3f | lr %.2e | grad %.2f | %.1fs",
                state.step,
                cfg.max_steps,
                window_loss / window_tokens,
                window_correct / window_tokens,
                lr,
                float(grad_norm),
                elapsed,
            )
            state.history.append(
                {
                    "step": state.step,
                    "train_loss": window_loss / window_tokens,
                    "train_accuracy": window_correct / window_tokens,
                    "lr": lr,
                }
            )
            window_loss = window_correct = window_tokens = 0.0

        if state.step % cfg.eval_every == 0 or state.step == cfg.max_steps:
            metrics = evaluate(model, val_stream, cfg, device, precision)
            logger.info(
                "eval step %d | val loss %.4f | val acc %.3f | val ppl %.2f",
                state.step,
                metrics["loss"],
                metrics["accuracy"],
                metrics["perplexity"],
            )
            state.history.append({"step": state.step, **{f"val_{k}": v for k, v in metrics.items()}})
            if metrics["loss"] < state.best_val_loss:
                state.best_val_loss = metrics["loss"]
                checkpoint(BEST_CHECKPOINT)
            checkpoint(LAST_CHECKPOINT)
            (out_dir / HISTORY_FILE).write_text(json.dumps(state.history, indent=2))

    plot_history(state.history, out_dir / "training_curves.png")
    return state


def plot_history(history: list[dict[str, Any]], path: Path) -> Path | None:
    """Loss and accuracy curves as a PNG (skipped if matplotlib is not installed)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    train = [h for h in history if "train_loss" in h]
    val = [h for h in history if "val_loss" in h]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in ((ax_loss, "loss", "Loss"), (ax_acc, "accuracy", "Next-token accuracy")):
        ax.plot([h["step"] for h in train], [h[f"train_{key}"] for h in train], label="train")
        ax.plot([h["step"] for h in val], [h[f"val_{key}"] for h in val], "o-", label="validation")
        ax.set(title=title, xlabel="step")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a music generator (Transformer, LSTM or RNN) on ABC tunes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=MODEL_TYPES, default="transformer")
    m = p.add_argument_group("model (unset = preset for --model, see config.py)")
    m.add_argument("--impl", choices=RECURRENT_IMPLS, help="rnn/lstm: hand-written cells or torch")
    m.add_argument("--activation", choices=ACTIVATIONS, help="rnn only")
    m.add_argument("--d-model", type=int, help="token embedding size")
    m.add_argument("--layers", type=int, dest="n_layers", help="blocks / stacked recurrent layers")
    m.add_argument("--head-dim", type=int, help="transformer: size of one attention head")
    m.add_argument("--ff-dim", type=int, help="transformer: MLP size (default 4 * d_model)")
    m.add_argument("--max-seq-len", type=int, help="transformer: context length")
    m.add_argument("--hidden-dim", type=int, help="rnn/lstm: hidden state size")
    m.add_argument("--dropout", type=float)
    m.add_argument("--bias", action=argparse.BooleanOptionalAction, default=None)

    d = p.add_argument_group("data")
    d.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    d.add_argument("--train-file", type=Path, help="JSON list of {'abc notation': ...}")
    d.add_argument("--val-file", type=Path)
    d.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="BPE .model file (trained automatically if missing)",
    )
    d.add_argument("--train-limit", type=int, help="use only the first N training tunes")
    d.add_argument("--val-limit", type=int)
    d.add_argument(
        "--overfit", type=int, metavar="N", help="sanity check: train AND validate on the first N tunes"
    )

    t = p.add_argument_group("training (unset = preset for --model)")
    t.add_argument("--output-dir", type=Path)
    t.add_argument("--max-steps", type=int)
    t.add_argument("--batch-size", type=int)
    t.add_argument("--seq-len", type=int, help="tokens per training window")
    t.add_argument("--grad-accum-steps", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--min-lr", type=float)
    t.add_argument("--warmup-steps", type=int)
    t.add_argument("--weight-decay", type=float)
    t.add_argument("--grad-clip", type=float)
    t.add_argument("--label-smoothing", type=float)
    t.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"])
    t.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    t.add_argument("--log-every", type=int)
    t.add_argument("--eval-every", type=int)
    t.add_argument("--eval-batches", type=int)
    t.add_argument("--seed", type=int)
    t.add_argument("--resume", type=Path, help="checkpoint to continue from (e.g. .../last.pt)")
    return p


def get_tokenizer(data_cfg: DataConfig, vocab_size: int = 856, sample_tunes: int = 20_000) -> MGTokenizer:
    """Load the tokenizer, or train one on a sample of the training set if it is missing."""
    path = Path(data_cfg.tokenizer_path)
    if path.exists():
        return MGTokenizer.load(path)
    logger.info("tokenizer %s not found: training one on %d tunes", path, sample_tunes)
    tunes = load_tunes(resolve_split_file(data_cfg, "train"), sample_tunes)
    tokenizer = MGTokenizer()
    tokenizer.train(format_corpus(tunes), vocab_size)
    tokenizer.save(path)
    return tokenizer


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        tokenizer_path=args.tokenizer,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )
    train_keys = (
        "output_dir",
        "max_steps",
        "batch_size",
        "seq_len",
        "grad_accum_steps",
        "lr",
        "min_lr",
        "warmup_steps",
        "weight_decay",
        "grad_clip",
        "label_smoothing",
        "precision",
        "device",
        "log_every",
        "eval_every",
        "eval_batches",
        "seed",
        "resume",
    )
    train_cfg = make_train_config(args.model, **{k: getattr(args, k) for k in train_keys})
    if args.output_dir is None:
        train_cfg = replace(train_cfg, output_dir=Path("checkpoints") / args.model)
    seed_everything(train_cfg.seed)

    tokenizer = get_tokenizer(data_cfg)
    model_keys = (
        "impl",
        "activation",
        "d_model",
        "n_layers",
        "head_dim",
        "ff_dim",
        "max_seq_len",
        "hidden_dim",
        "dropout",
        "bias",
    )
    model_cfg = make_model_config(
        args.model, tokenizer.vocab_size, **{k: getattr(args, k) for k in model_keys}
    )
    train_stream, val_stream = build_streams(data_cfg, tokenizer, overfit=args.overfit)
    model = build_model(model_cfg)
    state = train(
        model,
        tokenizer,
        train_stream,
        val_stream,
        train_cfg,
        extra_metadata={"data_config": config_to_dict(data_cfg)},
    )
    logger.info("done: best val loss %.4f, checkpoints in %s", state.best_val_loss, train_cfg.output_dir)


if __name__ == "__main__":
    main()
