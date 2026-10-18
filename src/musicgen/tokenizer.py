"""Byte-level BPE tokenizer for ABC notation (the original ``MGTokenizer``).

Text is first split into chunks with a GPT-4 style regex, then each chunk's
UTF-8 bytes are merged pair by pair in the order learned during training.
Every byte is in the base vocabulary, so any text can be encoded (no OOV).

Three special tokens frame and pad the tunes:

    <SOS> X:1 L:1/8 M:6/8 K:G ... |] <EOS>

The ``.model`` file format is unchanged from the original project
("minbpe v1"), so tokenizers trained with the legacy code load here too.

A character-level tokenizer (the original ``NaiveTokenizer``) is kept as a
simple baseline: one id per distinct character.
"""

from __future__ import annotations

import itertools
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import regex as re

SOS, EOS, PAD = "<SOS>", "<EOS>", "<PAD>"
SPECIAL_TOKENS = (PAD, SOS, EOS)

GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

Pair = tuple[int, int]


def get_stats(ids: list[int], counts: dict[Pair, int] | None = None, weight: int = 1) -> dict[Pair, int]:
    """Count consecutive pairs in ``ids`` (each occurrence counts ``weight`` times)."""
    counts = {} if counts is None else counts
    for pair in itertools.pairwise(ids):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def merge(ids: list[int], pair: Pair, idx: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with the new token ``idx``."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def render_token(token: bytes) -> str:
    """Printable version of a token (control characters escaped) for .vocab files."""
    text = token.decode("utf-8", errors="replace")
    return "".join(ch if unicodedata.category(ch)[0] != "C" else f"\\u{ord(ch):04x}" for ch in text)


class MGTokenizer:
    """Regex + byte-level BPE tokenizer with special tokens."""

    def __init__(self, pattern: str | None = None) -> None:
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges: dict[Pair, int] = {}
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}
        self.vocab: dict[int, bytes] = self._build_vocab()
        self._cache: dict[str, list[int]] = {}

    # -- properties ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Embedding table size: highest id + 1 (legacy files leave gaps in the ids)."""
        ids = [*self.vocab, *self.special_tokens.values()]
        return max(ids) + 1

    @property
    def sos_id(self) -> int:
        return self.special_tokens[SOS]

    @property
    def eos_id(self) -> int:
        return self.special_tokens[EOS]

    @property
    def pad_id(self) -> int:
        return self.special_tokens[PAD]

    # -- training -----------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn ``vocab_size - 256`` merges on ``text``, then add the special tokens.

        Identical chunks are counted once with a weight (ABC tunes repeat the
        same bars a lot), which makes training many times faster than the
        original loop while learning the same kind of merges.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte alphabet)")
        num_merges = vocab_size - 256
        chunk_counts = Counter(re.findall(self.compiled_pattern, text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        weights = list(chunk_counts.values())

        merges: dict[Pair, int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats: dict[Pair, int] = {}
            for ids, weight in zip(words, weights):
                get_stats(ids, stats, weight)
            if not stats:  # every chunk is a single token already
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            words = [merge(ids, pair, idx) if len(ids) > 1 else ids for ids in words]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"{time.strftime('%H:%M:%S')} merge {i + 1}/{num_merges}: "
                    f"{pair} -> {idx} ({render_token(vocab[idx])!r}, {stats[pair]} times)",
                    flush=True,
                )

        self.merges = merges
        first_special = 256 + len(merges)
        self.register_special_tokens({tok: first_special + i for i, tok in enumerate(SPECIAL_TOKENS)})

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        self.special_tokens = dict(special_tokens)
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self.vocab = self._build_vocab()
        self._cache.clear()

    # -- encode / decode ----------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Text -> ids. Special tokens written in the text become single ids."""
        if not self.special_tokens:
            return self._encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        ids: list[int] = []
        for part in re.split(special_pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[bytes] = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            elif idx in self.vocab:
                parts.append(self.vocab[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
        return b"".join(parts).decode("utf-8", errors="replace")

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in re.findall(self.compiled_pattern, text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk.encode("utf-8"))
                if len(self._cache) < 1_000_000:
                    self._cache[chunk] = cached
            ids.extend(cached)
        return ids

    def _encode_chunk(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    # -- save / load --------------------------------------------------------

    def save(self, file_prefix: str | Path) -> Path:
        """Write ``<prefix>.model`` (loadable) and ``<prefix>.vocab`` (human-readable)."""
        prefix = str(file_prefix).removesuffix(".model")
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        model_file = Path(prefix + ".model")
        with model_file.open("w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        inverted = {idx: pair for pair, idx in self.merges.items()}
        with open(prefix + ".vocab", "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                text = render_token(token)
                if idx in inverted:
                    left, right = (render_token(self.vocab[i]) for i in inverted[idx])
                    f.write(f"[{left}][{right}] -> [{text}] {idx}\n")
                else:
                    f.write(f"[{text}] {idx}\n")
        return model_file

    @classmethod
    def load(cls, model_file: str | Path) -> MGTokenizer:
        model_file = Path(model_file)
        if model_file.suffix != ".model":
            raise ValueError(f"expected a .model file, got {model_file}")
        with model_file.open(encoding="utf-8") as f:
            if f.readline().strip() != "minbpe v1":
                raise ValueError(f"{model_file} is not a minbpe v1 tokenizer file")
            tokenizer = cls(pattern=f.readline().rstrip("\n"))
            special_tokens = {}
            for _ in range(int(f.readline())):
                special, idx = f.readline().split()
                special_tokens[special] = int(idx)
            merges = {}
            for idx, line in enumerate(f, start=256):
                left, right = map(int, line.split())
                merges[(left, right)] = idx
        tokenizer.merges = merges
        tokenizer.register_special_tokens(special_tokens)
        return tokenizer

    def to_dict(self) -> dict:
        """Everything needed to rebuild the tokenizer (stored inside checkpoints)."""
        return {
            "pattern": self.pattern,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MGTokenizer:
        tokenizer = cls(pattern=data["pattern"])
        tokenizer.merges = {(a, b): idx for a, b, idx in data["merges"]}
        tokenizer.register_special_tokens(data["special_tokens"])
        return tokenizer

    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab


class CharTokenizer:
    """One id per distinct character of the training text (the original NaiveTokenizer).

    Characters never seen in training are dropped by ``encode``.
    """

    def __init__(self, text: str, extra_tokens: Iterable[str] = ()) -> None:
        self.tokens = sorted(set(text))
        self.stoi = {s: i for i, s in enumerate(self.tokens)}
        for token in extra_tokens:
            self.stoi.setdefault(token, len(self.stoi))
        self.itos = {i: s for s, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)
