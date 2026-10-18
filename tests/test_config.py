import pytest

from musicgen.config import (
    MODEL_PRESETS,
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_to_dict,
    make_model_config,
    make_train_config,
    model_config_from_dict,
)


def test_presets_and_overrides():
    cfg = make_model_config("lstm", 100, hidden_dim=64, dropout=None)
    assert cfg.hidden_dim == 64  # override wins
    assert cfg.d_model == MODEL_PRESETS["lstm"]["d_model"]  # preset kept
    assert cfg.dropout == 0.0  # None means "not set"
    train = make_train_config("transformer", lr=None, max_steps=7)
    assert train.lr == 2e-4 and train.seq_len == 512 and train.max_steps == 7
    assert make_train_config("rnn").seq_len == 200


def test_transformer_heads():
    cfg = make_model_config("transformer", 100, d_model=128, head_dim=32)
    assert cfg.n_heads == 4 and cfg.ffn_dim == 512


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_type": "gru"},
        {"vocab_size": 0},
        {"dropout": 1.0},
        {"d_model": 100, "head_dim": 32},  # not divisible
        {"impl": "cuda"},
        {"activation": "gelu"},
    ],
)
def test_invalid_model_config(kwargs):
    values = {"model_type": "transformer", "vocab_size": 10, **kwargs}
    with pytest.raises(ValueError):
        ModelConfig(**values)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_steps": 0}, {"lr": 1e-4, "min_lr": 1e-3}, {"precision": "int8"}, {"device": "tpu"}],
)
def test_invalid_train_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_invalid_data_config():
    with pytest.raises(ValueError):
        DataConfig(train_limit=0)


def test_config_roundtrip():
    cfg = make_model_config("rnn", 50, activation="relu")
    assert model_config_from_dict(config_to_dict(cfg)) == cfg
    assert isinstance(config_to_dict(TrainConfig())["output_dir"], str)
