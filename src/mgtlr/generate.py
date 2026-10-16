"""Generate tunes with a trained checkpoint.

    mgtlr-generate --checkpoint checkpoints/transformer/best.pt --num-tunes 3 --midi

The prompt is the start of an ABC tune (often just its header); the model
continues it until it writes <EOS> or reaches --max-new-tokens. Each tune is
saved as ``.abc`` and, with --midi (needs music21), as ``.mid``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from mgtlr.checkpoint import load_checkpoint
from mgtlr.models import LanguageModel
from mgtlr.music import abc_to_midi, extract_tunes
from mgtlr.tokenizer import SOS, MGTokenizer
from mgtlr.train import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "X:1\nL:1/8\nM:6/8\nK:G\n"


def generate_abc(
    model: LanguageModel,
    tokenizer: MGTokenizer,
    prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    keep_unfinished: bool = True,
) -> str | None:
    """Continue ``prompt`` into one tune. Returns the ABC text, or None if empty."""
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(SOS + prompt)], device=device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    out = model.generate(
        ids, max_new_tokens, temperature, top_k or None, eos_id=tokenizer.eos_id, generator=generator
    )
    tunes = extract_tunes(tokenizer.decode(out[0].tolist()), keep_unfinished=keep_unfinished)
    return tunes[0] if tunes else None


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, help="only for checkpoints from the legacy code")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="start of the tune (ABC); '\\n' allowed")
    p.add_argument("--prompt-file", type=Path, help="read the prompt from a file instead")
    p.add_argument("--num-tunes", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.9, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=0, help="0 = no top-k filtering")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--midi", action="store_true", help="also write .mid files (needs music21)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt.replace("\\n", "\n")
    loaded = load_checkpoint(args.checkpoint, resolve_device(args.device), args.tokenizer)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_tunes):
        seed = None if args.seed is None else args.seed + i
        abc = generate_abc(
            loaded.model, loaded.tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k, seed
        )
        if abc is None:
            logger.warning("tune %d: the model produced nothing", i + 1)
            continue
        abc_path = args.out_dir / f"tune_{i + 1:02d}.abc"
        abc_path.write_text(abc + "\n", encoding="utf-8")
        print(f"--- {abc_path}\n{abc}\n")
        if args.midi:
            try:
                logger.info("wrote %s", abc_to_midi(abc, abc_path.with_suffix(".mid")))
            except Exception as exc:  # noqa: BLE001 - music21 raises many error types on bad ABC
                logger.warning("tune %d: not valid enough for MIDI (%s)", i + 1, exc)


if __name__ == "__main__":
    main()
