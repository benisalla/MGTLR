"""Train the BPE tokenizer on ABC tunes.

    musicgen-tokenizer --vocab-size 856 --sample-tunes 50000

Writes ``<output>.model`` (used by training) and ``<output>.vocab`` (readable).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from musicgen.config import DEFAULT_DATA_DIR, DEFAULT_TOKENIZER_PATH, DataConfig
from musicgen.data import format_corpus, load_tunes, resolve_split_file
from musicgen.tokenizer import SOS, MGTokenizer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--train-file", type=Path, help="JSON list of {'abc notation': ...}")
    p.add_argument("--vocab-size", type=int, default=856, help="256 bytes + merges (+3 specials)")
    p.add_argument("--sample-tunes", type=int, default=20_000, help="tunes used for training")
    p.add_argument("--output", type=Path, default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--verbose", action="store_true", help="print every merge")
    args = p.parse_args(argv)

    data_cfg = DataConfig(data_dir=args.data_dir, train_file=args.train_file)
    tunes = load_tunes(resolve_split_file(data_cfg, "train"), args.sample_tunes)
    text = format_corpus(tunes)
    print(f"training on {len(tunes):,} tunes ({len(text):,} characters)")
    start = time.perf_counter()
    tokenizer = MGTokenizer()
    tokenizer.train(text, args.vocab_size, verbose=args.verbose)
    path = tokenizer.save(args.output)
    print(f"saved {path} (vocab size {tokenizer.vocab_size}) in {time.perf_counter() - start:.0f}s")

    sample = SOS + tunes[0][:80]
    ids = tokenizer.encode(sample)
    print(f"{len(sample)} characters -> {len(ids)} tokens; round trip ok: {tokenizer.decode(ids) == sample}")


if __name__ == "__main__":
    main()
