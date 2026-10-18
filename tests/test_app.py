"""Headless smoke test of the Streamlit app (skipped if streamlit is not installed)."""

from pathlib import Path

import pytest

from musicgen.checkpoint import save_checkpoint
from musicgen.config import make_model_config
from musicgen.models import build_model

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_app_generates_and_reports_errors(tmp_path, tokenizer):
    model = build_model(
        make_model_config("lstm", tokenizer.vocab_size, d_model=8, hidden_dim=8, n_layers=1, impl="torch")
    )
    checkpoint = save_checkpoint(tmp_path / "tiny.pt", model, tokenizer)

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert at.sidebar.radio[0].options == ["TRF", "LSTM", "RNN"]

    at.sidebar.radio[0].set_value("LSTM").run()
    at.sidebar.text_input[0].set_value(str(tmp_path / "missing.pt")).run()
    at.sidebar.button[0].click().run()
    assert any("No checkpoint" in e.value for e in at.error)

    # an untrained model writes invalid ABC: the app must show its error card, not crash
    at.sidebar.text_input[0].set_value(str(checkpoint)).run()
    at.sidebar.slider[0].set_value(256).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.session_state["songs"] == [] or at.session_state["songs"][0]["midi"]
