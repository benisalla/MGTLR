"""Shared fixtures: a few real-looking ABC tunes and a tiny BPE tokenizer (CPU, offline)."""

import json

import pytest

from mgtlr.data import format_corpus
from mgtlr.tokenizer import MGTokenizer

TUNES = [
    "X:1\nL:1/8\nM:6/8\nK:G\nGAB cBA | BGE EDE | GAB cBA | BGE E3 :|",
    "X:2\nL:1/8\nM:4/4\nK:D\nA2FA DAFA | BcdB AFED | A2FA DAFA | BcdB A4 |]",
    "X:3\nL:1/8\nM:3/4\nK:Amin\nE2 A2 c2 | B2 G2 E2 | A2 c2 e2 | d6 |]",
    "X:4\nL:1/8\nM:2/4\nK:C\nc2 e2 | g2 e2 | f2 d2 | c4 :|",
]


@pytest.fixture(scope="session")
def tokenizer() -> MGTokenizer:
    tok = MGTokenizer()
    tok.train(format_corpus(TUNES * 3), vocab_size=300)
    return tok


@pytest.fixture
def dataset_dir(tmp_path):
    """data_dir with train.json / validation.json in the IrishMAN format."""
    records = [{"control code": "S:2", "abc notation": t} for t in TUNES]
    (tmp_path / "train.json").write_text(json.dumps(records * 5))
    (tmp_path / "validation.json").write_text(json.dumps(records[:2] * 3))
    return tmp_path
