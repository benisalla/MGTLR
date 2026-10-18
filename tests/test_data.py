import pytest
import torch

from musicgen.config import DataConfig
from musicgen.data import TokenStream, build_streams, encode_corpus, format_corpus, load_tunes
from musicgen.tokenizer import EOS, SOS
from tests.conftest import TUNES


def test_load_tunes_and_limit(dataset_dir):
    tunes = load_tunes(dataset_dir / "train.json", limit=3)
    assert tunes == TUNES[:3]


def test_format_corpus_frames_every_tune():
    text = format_corpus(TUNES[:2])
    assert text == f"{SOS}{TUNES[0]}{EOS}\n\n{SOS}{TUNES[1]}{EOS}"


def test_batches_are_shifted_windows():
    stream = TokenStream(torch.arange(100))
    x, y = stream.sample_batch(4, 10, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 10)
    assert torch.equal(y, x + 1)  # target = next token
    assert torch.equal(x[:, 1:], x[:, :-1] + 1)  # contiguous window
    assert int(y.max()) <= 99


def test_batches_are_reproducible():
    stream = TokenStream(torch.randint(0, 50, (500,)))
    a = stream.sample_batch(3, 8, torch.Generator().manual_seed(1))
    b = stream.sample_batch(3, 8, torch.Generator().manual_seed(1))
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_stream_too_short():
    with pytest.raises(ValueError):
        TokenStream(torch.arange(5)).sample_batch(1, 5)


def test_encode_cache(tokenizer, tmp_path):
    text = format_corpus(TUNES)
    first = encode_corpus(text, tokenizer, tmp_path)
    assert len(list(tmp_path.glob("tokens-*.npy"))) == 1
    second = encode_corpus(text, tokenizer, tmp_path)
    assert torch.equal(first, second) and second.dtype == torch.long
    assert tokenizer.decode(first.tolist()) == text


def test_build_streams(dataset_dir, tokenizer):
    cfg = DataConfig(data_dir=dataset_dir)
    train, val = build_streams(cfg, tokenizer)
    assert len(train) > len(val) > 0
    over_train, over_val = build_streams(cfg, tokenizer, overfit=2)
    assert torch.equal(over_train.tokens, over_val.tokens)  # same tunes on both sides
    assert tokenizer.decode(over_train.tokens.tolist()) == format_corpus(TUNES[:2])
