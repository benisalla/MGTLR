"""Checkpoints, legacy compatibility, the generate CLI and ABC -> MIDI."""

import sys
from pathlib import Path

import pytest
import torch

from mgtlr.checkpoint import load_checkpoint, save_checkpoint
from mgtlr.config import make_model_config
from mgtlr.generate import generate_abc, main
from mgtlr.models import build_model
from mgtlr.music import abc_to_midi, extract_tunes
from mgtlr.tokenizer import EOS, PAD, SOS
from tests.conftest import TUNES

LEGACY_DIR = Path(__file__).resolve().parents[1] / "legacy"


@pytest.fixture
def checkpoint(tmp_path, tokenizer):
    torch.manual_seed(0)
    model = build_model(
        make_model_config(
            "transformer", tokenizer.vocab_size, d_model=16, head_dim=8, n_layers=1, max_seq_len=64
        )
    )
    return save_checkpoint(tmp_path / "model.pt", model, tokenizer, step=3), model


def test_checkpoint_roundtrip(checkpoint, tokenizer):
    path, model = checkpoint
    loaded = load_checkpoint(path)
    assert loaded.step == 3
    assert loaded.tokenizer.encode(TUNES[0]) == tokenizer.encode(TUNES[0])
    x = torch.randint(0, tokenizer.vocab_size, (1, 10))
    model.eval()
    with torch.no_grad():
        assert torch.equal(loaded.model(x).logits, model(x).logits)


def test_loads_checkpoint_from_the_legacy_code(tmp_path, tokenizer, monkeypatch):
    """A checkpoint saved by legacy/music_generator (MGTransformer + save_checkpoints) loads."""
    monkeypatch.syspath_prepend(str(LEGACY_DIR))
    from music_generator.model.TRF.MGTransformer import MGTransformer as LegacyTransformer

    torch.manual_seed(0)
    legacy = LegacyTransformer(
        n_embd=32, h_dim=16, n_block=2, v_size=tokenizer.vocab_size, b_size=8, max_seq_len=64
    ).eval()
    path = tmp_path / "mg_chpts_v1.pth"
    torch.save({"model": legacy.state_dict(), "model_args": legacy.get_init_args(), "epoch": 1}, path)
    tok_path = tokenizer.save(tmp_path / "mgt_tokenizer_v1")

    loaded = load_checkpoint(path, tokenizer_path=tok_path)
    x = torch.randint(0, tokenizer.vocab_size, (2, 12))
    with torch.no_grad():
        old_logits, _ = legacy(x, x)
        assert torch.allclose(loaded.model(x).logits, old_logits, atol=1e-5)
    with pytest.raises(ValueError):
        load_checkpoint(path)  # legacy files need the separate tokenizer
    for name in list(sys.modules):
        if name.startswith("music_generator"):
            del sys.modules[name]


def test_extract_tunes():
    text = f"{SOS}X:1\nK:G\nabc|{EOS}\n\n{SOS}X:2\nK:D\nde{PAD}f|{EOS}{SOS}X:3\nK:C\ncut"
    assert extract_tunes(text) == ["X:1\nK:G\nabc|", "X:2\nK:D\ndef|"]
    assert extract_tunes(text, keep_unfinished=True)[-1] == "X:3\nK:C\ncut"
    assert extract_tunes("no tunes here") == []
    # a new <SOS> before any <EOS> cuts the previous tune
    assert extract_tunes(f"{SOS}ab{SOS}cd{EOS}", keep_unfinished=True) == ["ab", "cd"]
    assert extract_tunes(f"{SOS}ab{SOS}cd{EOS}") == ["cd"]


def test_generate_abc_is_reproducible(checkpoint, tokenizer):
    _, model = checkpoint
    a = generate_abc(model, tokenizer, "X:1\n", max_new_tokens=20, seed=3)
    b = generate_abc(model, tokenizer, "X:1\n", max_new_tokens=20, seed=3)
    assert a == b and a.startswith("X:1")


def test_generate_cli_writes_abc(checkpoint, tmp_path):
    path, _ = checkpoint
    out = tmp_path / "out"
    main(
        [
            "--checkpoint",
            str(path),
            "--num-tunes",
            "2",
            "--max-new-tokens",
            "15",
            "--seed",
            "0",
            "--out-dir",
            str(out),
            "--device",
            "cpu",
            "--prompt",
            "X:1\\nK:G\\n",
        ]
    )
    files = sorted(out.glob("*.abc"))
    assert [f.name for f in files] == ["tune_01.abc", "tune_02.abc"]
    text = files[0].read_text()
    assert text.startswith("X:1\nK:G\n") and SOS not in text and EOS not in text


def test_abc_to_midi(tmp_path):
    pytest.importorskip("music21")
    midi = abc_to_midi(TUNES[0], tmp_path / "tune.mid")
    assert midi.read_bytes()[:4] == b"MThd"  # standard MIDI file header
