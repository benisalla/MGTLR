"""ABC tunes -> one long token stream -> random training windows.

The corpus is the IrishMAN dataset: JSON lists of ``{"abc notation": ...}``
records (``train.json`` and ``validation.json``, downloaded on first use).
Every tune is framed with special tokens and the tunes are joined into one
text, exactly like the original project::

    <SOS>X:1\\nL:1/8\\nM:6/8\\nK:G\\n...|]<EOS>\\n\\n<SOS>X:2 ...

Training samples are random windows of ``seq_len + 1`` tokens taken from that
stream: the model reads tokens ``[i, i + seq_len)`` and predicts
``[i + 1, i + seq_len + 1)``. A window can cross tune boundaries, which is how
the model learns to emit ``<EOS>`` and start a new tune.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from musicgen.config import DataConfig
from musicgen.tokenizer import EOS, SOS, MGTokenizer

logger = logging.getLogger(__name__)

ABC_FIELD = "abc notation"
TUNE_SEPARATOR = "\n\n"
SPLIT_FILES = {"train": "train.json", "validation": "validation.json"}


def download_split(split: str, data_dir: Path, dataset_id: str = "sander-wood/irishman") -> Path:
    """Fetch ``train.json`` / ``validation.json`` from the Hugging Face Hub (once)."""
    path = Path(data_dir) / SPLIT_FILES[split]
    if path.exists():
        return path
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{SPLIT_FILES[split]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s -> %s", url, path)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.rename(path)
    return path


def resolve_split_file(config: DataConfig, split: str) -> Path:
    explicit = config.train_file if split == "train" else config.val_file
    if explicit is not None:
        return Path(explicit)
    return download_split(split, config.data_dir, config.dataset_id)


def load_tunes(path: str | Path, limit: int | None = None) -> list[str]:
    """Read a JSON list of records and return the non-empty ABC strings."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    tunes = [r[ABC_FIELD].strip() for r in records if r.get(ABC_FIELD, "").strip()]
    return tunes[:limit] if limit is not None else tunes


def format_corpus(tunes: Sequence[str]) -> str:
    """Frame every tune with <SOS>/<EOS> and join them (the training text)."""
    return TUNE_SEPARATOR.join(f"{SOS}{tune}{EOS}" for tune in tunes)


def encode_corpus(text: str, tokenizer: MGTokenizer, cache_dir: Path | None = None) -> torch.Tensor:
    """Tokenize the corpus, caching the result as .npy (encoding 60M chars takes minutes)."""
    cache_file = None
    if cache_dir is not None:
        key = hashlib.sha1()
        key.update(json.dumps(tokenizer.to_dict(), sort_keys=True).encode())
        key.update(text.encode("utf-8"))
        cache_file = Path(cache_dir) / f"tokens-{key.hexdigest()[:16]}.npy"
        if cache_file.exists():
            return torch.from_numpy(np.load(cache_file).astype(np.int64))
    ids = np.asarray(tokenizer.encode(text), dtype=np.int32)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, ids)
    return torch.from_numpy(ids.astype(np.int64))


class TokenStream:
    """A 1-D tensor of token ids that serves random (input, target) windows."""

    def __init__(self, tokens: torch.Tensor) -> None:
        if tokens.dim() != 1:
            raise ValueError("tokens must be a 1-D tensor")
        self.tokens = tokens.long()

    def __len__(self) -> int:
        return self.tokens.numel()

    def sample_batch(
        self, batch_size: int, seq_len: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two ``[batch_size, seq_len]`` tensors; targets are inputs shifted by one."""
        if len(self) <= seq_len:
            raise ValueError(f"stream has {len(self)} tokens, need more than seq_len={seq_len}")
        starts = torch.randint(0, len(self) - seq_len, (batch_size,), generator=generator)
        offsets = torch.arange(seq_len + 1)
        windows = self.tokens[starts[:, None] + offsets]  # [B, seq_len + 1]
        return windows[:, :-1], windows[:, 1:]


def build_streams(
    config: DataConfig, tokenizer: MGTokenizer, overfit: int | None = None
) -> tuple[TokenStream, TokenStream]:
    """Load, frame and tokenize both splits. ``overfit=N`` uses N train tunes for both."""
    train_tunes = load_tunes(resolve_split_file(config, "train"), overfit or config.train_limit)
    if overfit:
        val_tunes = train_tunes
    else:
        val_tunes = load_tunes(resolve_split_file(config, "validation"), config.val_limit)
    cache_dir = Path(config.data_dir) / "cache"
    train = TokenStream(encode_corpus(format_corpus(train_tunes), tokenizer, cache_dir))
    val = TokenStream(encode_corpus(format_corpus(val_tunes), tokenizer, cache_dir))
    logger.info(
        "data: %d train tunes (%d tokens), %d val tunes (%d tokens)",
        len(train_tunes),
        len(train),
        len(val_tunes),
        len(val),
    )
    return train, val
