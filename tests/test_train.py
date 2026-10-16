import json
from dataclasses import replace

import pytest
import torch

from mgtlr.checkpoint import load_checkpoint
from mgtlr.config import TrainConfig, make_model_config, make_train_config
from mgtlr.data import TokenStream, encode_corpus, format_corpus
from mgtlr.models import build_model
from mgtlr.train import (
    BEST_CHECKPOINT,
    HISTORY_FILE,
    LAST_CHECKPOINT,
    evaluate,
    lr_at,
    main,
    make_optimizer,
    train,
)
from tests.conftest import TUNES


def tiny_cfg(tmp_path, **overrides) -> TrainConfig:
    values = {
        "output_dir": tmp_path / "run",
        "max_steps": 30,
        "batch_size": 8,
        "seq_len": 16,
        "lr": 3e-3,
        "warmup_steps": 3,
        "log_every": 10,
        "eval_every": 15,
        "eval_batches": 2,
        "device": "cpu",
    }
    return replace(make_train_config("transformer"), **{**values, **overrides})


def stream(tokenizer) -> TokenStream:
    return TokenStream(encode_corpus(format_corpus(TUNES * 4), tokenizer))


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-5, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == pytest.approx(1e-4)
    assert lr_at(9, cfg) == pytest.approx(1e-3)
    assert lr_at(60, cfg) == pytest.approx((1e-3 + 1e-5) / 2)
    assert lr_at(110, cfg) == pytest.approx(1e-5)
    assert lr_at(500, cfg) == pytest.approx(1e-5)  # stays at min after the end


def test_optimizer_groups(tokenizer):
    model = build_model(make_model_config("transformer", tokenizer.vocab_size, d_model=16, head_dim=8))
    opt = make_optimizer(model, TrainConfig())
    decayed, plain = opt.param_groups
    n_params = sum(1 for _ in model.parameters())
    assert len(decayed["params"]) + len(plain["params"]) == n_params  # tied weight counted once
    assert all(p.dim() >= 2 for p in decayed["params"]) and all(p.dim() < 2 for p in plain["params"])


@pytest.mark.parametrize("model_type", ["transformer", "lstm", "rnn"])
def test_training_lowers_loss_and_writes_checkpoints(tmp_path, tokenizer, model_type):
    torch.manual_seed(0)
    model_cfg = make_model_config(
        model_type, tokenizer.vocab_size, d_model=32, head_dim=16, n_layers=1, hidden_dim=48, max_seq_len=32
    )
    model = build_model(model_cfg)
    data = stream(tokenizer)
    cfg = tiny_cfg(tmp_path)
    before = evaluate(model, data, cfg, torch.device("cpu"))["loss"]
    state = train(model, tokenizer, data, data, cfg)
    assert state.step == 30 and state.best_val_loss < 0.8 * before
    out = cfg.output_dir
    for name in (LAST_CHECKPOINT, BEST_CHECKPOINT, HISTORY_FILE, "training_curves.png"):
        assert (out / name).exists()
    history = json.loads((out / HISTORY_FILE).read_text())
    assert [h["step"] for h in history if "val_loss" in h] == [15, 30]

    loaded = load_checkpoint(out / LAST_CHECKPOINT)
    assert loaded.step == 30 and loaded.model.config == model_cfg
    x, _ = data.sample_batch(2, 16, torch.Generator().manual_seed(0))
    model.eval()
    with torch.no_grad():
        assert torch.allclose(loaded.model(x).logits, model(x).logits, atol=1e-6)


def test_resume_continues(tmp_path, tokenizer):
    data = stream(tokenizer)
    model_cfg = make_model_config("lstm", tokenizer.vocab_size, d_model=16, hidden_dim=16, n_layers=1)
    cfg = tiny_cfg(tmp_path, max_steps=10, eval_every=10)
    train(build_model(model_cfg), tokenizer, data, data, cfg)
    resumed = train(
        build_model(model_cfg),
        tokenizer,
        data,
        data,
        replace(cfg, max_steps=20, resume=cfg.output_dir / LAST_CHECKPOINT),
    )
    assert resumed.step == 20
    assert [h["step"] for h in resumed.history if "val_loss" in h] == [10, 20]


def test_grad_accumulation_counts_optimizer_steps(tmp_path, tokenizer):
    data = stream(tokenizer)
    model = build_model(make_model_config("rnn", tokenizer.vocab_size, d_model=16, hidden_dim=16))
    state = train(model, tokenizer, data, data, tiny_cfg(tmp_path, max_steps=4, grad_accum_steps=3))
    assert state.step == 4


def test_seq_len_must_fit_transformer(tmp_path, tokenizer):
    model = build_model(make_model_config("transformer", tokenizer.vocab_size, max_seq_len=8))
    with pytest.raises(ValueError):
        train(model, tokenizer, stream(tokenizer), stream(tokenizer), tiny_cfg(tmp_path, seq_len=16))


def test_cli_end_to_end(tmp_path, dataset_dir, tokenizer):
    """mgtlr-train with an existing tokenizer file and the --overfit sanity mode."""
    tok_path = tokenizer.save(tmp_path / "tok")
    out = tmp_path / "cli"
    main(
        [
            "--model",
            "lstm",
            "--impl",
            "torch",
            "--data-dir",
            str(dataset_dir),
            "--tokenizer",
            str(tok_path),
            "--overfit",
            "4",
            "--output-dir",
            str(out),
            "--max-steps",
            "6",
            "--batch-size",
            "4",
            "--seq-len",
            "16",
            "--eval-every",
            "3",
            "--eval-batches",
            "1",
            "--log-every",
            "3",
            "--d-model",
            "16",
            "--hidden-dim",
            "16",
            "--device",
            "cpu",
        ]
    )
    raw = load_checkpoint(out / LAST_CHECKPOINT).raw
    assert raw["step"] == 6 and raw["model_config"]["impl"] == "torch"
    assert raw["train_config"]["seq_len"] == 16


def test_cli_trains_missing_tokenizer(tmp_path, dataset_dir):
    tok_path = tmp_path / "new" / "tok.model"
    main(
        [
            "--model",
            "rnn",
            "--data-dir",
            str(dataset_dir),
            "--tokenizer",
            str(tok_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--seq-len",
            "8",
            "--eval-every",
            "2",
            "--eval-batches",
            "1",
            "--d-model",
            "8",
            "--hidden-dim",
            "8",
            "--device",
            "cpu",
        ]
    )
    assert tok_path.exists()
