import pytest

from mgtlr.tokenizer import EOS, PAD, SOS, CharTokenizer, MGTokenizer
from tests.conftest import TUNES


def test_roundtrip_and_compression(tokenizer):
    text = SOS + TUNES[0] + EOS
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert len(ids) < len(text)  # merges actually compress


def test_unseen_text_is_still_encodable(tokenizer):
    text = "X:9\nK:F#m\n^f'' ü ♪ |]"  # characters never seen in training
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_single_ids(tokenizer):
    ids = tokenizer.encode(f"{SOS}abc{EOS}")
    assert ids[0] == tokenizer.sos_id and ids[-1] == tokenizer.eos_id
    assert len({tokenizer.sos_id, tokenizer.eos_id, tokenizer.pad_id}) == 3
    assert tokenizer.vocab_size == 300 + 3
    assert max(tokenizer.encode(TUNES[1])) < tokenizer.vocab_size


def test_save_load_roundtrip(tokenizer, tmp_path):
    path = tokenizer.save(tmp_path / "tok")
    assert path.name == "tok.model" and (tmp_path / "tok.vocab").exists()
    loaded = MGTokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.special_tokens == tokenizer.special_tokens
    assert loaded.encode(TUNES[2]) == tokenizer.encode(TUNES[2])
    assert MGTokenizer.from_dict(tokenizer.to_dict()).encode(TUNES[3]) == tokenizer.encode(TUNES[3])


def test_loads_legacy_file_with_gaps(tokenizer, tmp_path):
    """The original project numbered specials vocab_size+1.. leaving a hole (and an <OOV>)."""
    n = 256 + len(tokenizer.merges)
    legacy = {PAD: n + 1, EOS: n + 2, SOS: n + 3, "<OOV>": n + 4}
    lines = ["minbpe v1", tokenizer.pattern, str(len(legacy))]
    lines += [f"{k} {v}" for k, v in legacy.items()]
    lines += [f"{a} {b}" for a, b in tokenizer.merges]
    path = tmp_path / "mgt_tokenizer_v1.model"
    path.write_text("\n".join(lines) + "\n")
    loaded = MGTokenizer.load(path)
    assert loaded.vocab_size == n + 5  # max id + 1, not len(vocab)
    assert loaded.encode(SOS)[0] == n + 3


def test_load_rejects_other_files(tmp_path):
    with pytest.raises(ValueError):
        MGTokenizer.load(tmp_path / "tok.json")
    bad = tmp_path / "bad.model"
    bad.write_text("something else\n")
    with pytest.raises(ValueError):
        MGTokenizer.load(bad)


def test_train_needs_byte_alphabet():
    with pytest.raises(ValueError):
        MGTokenizer().train("abc", vocab_size=100)


def test_char_tokenizer():
    tok = CharTokenizer("abca", extra_tokens=["%"])
    assert tok.vocab_size == 4
    assert tok.decode(tok.encode("cab%z")) == "cab%"  # unknown "z" dropped
