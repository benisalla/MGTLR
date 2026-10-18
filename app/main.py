"""Streamlit interface: pick a model, write the start of a tune, listen to the result.

    pip install -e '.[app]'
    streamlit run app/main.py

Checkpoints are looked up in ``checkpoints/<model>/best.pt`` (change the paths
in the sidebar). Audio needs the ``fluidsynth`` program and a SoundFont; without
them the app still shows the ABC and offers the MIDI file.
"""

from __future__ import annotations

import base64
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .` too
sys.path.insert(0, str(APP_DIR))

from components import ERROR_MESSAGE, SMILE_SPINNER

from musicgen.checkpoint import load_checkpoint
from musicgen.generate import generate_abc
from musicgen.music import abc_to_midi, midi_to_wav, wav_to_mp3

MODELS = {"TRF": "transformer", "LSTM": "lstm", "RNN": "rnn"}
PLACEHOLDER = "X:1\nT:BenIsAlla Tune\nM:3/4\nL:1/8\nK:Amin\nA,B,CE | DEFA | BAGE | A3z |"


@st.cache_resource(show_spinner=False)
def get_model(checkpoint: str, tokenizer: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_checkpoint(checkpoint, device, tokenizer or None)


def add_style() -> None:
    image = base64.b64encode((APP_DIR / "assets" / "music-bkgd.jpeg").read_bytes()).decode()
    css = (APP_DIR / "style.css").read_text().replace("{encoded_image}", image)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_song(song: dict, index: int) -> None:
    with st.container(border=True):
        if song.get("mp3"):
            st.audio(song["mp3"], format="audio/mp3")
            st.download_button(
                "Download MP3", song["mp3"], f"{song['name']}.mp3", "audio/mpeg", key=f"mp3-{index}"
            )
        if song.get("midi"):
            st.download_button(
                "Download MIDI", song["midi"], f"{song['name']}.mid", "audio/midi", key=f"midi-{index}"
            )
        with st.expander("Show ABC Annotations", expanded=song.get("mp3") is None):
            st.code(song["abc"], language=None)


def synthesize(abc: str, name: str) -> dict:
    """ABC -> MIDI -> MP3 in a temporary folder; keep the bytes in the session."""
    song = {"name": name, "abc": abc, "midi": None, "mp3": None}
    with tempfile.TemporaryDirectory() as tmp:
        midi = abc_to_midi(abc, Path(tmp) / f"{name}.mid")  # raises on invalid ABC
        song["midi"] = midi.read_bytes()
        try:
            song["mp3"] = wav_to_mp3(midi_to_wav(midi)).read_bytes()
        except Exception as exc:  # noqa: BLE001 - audio is optional, MIDI is enough
            st.toast(f"Audio unavailable ({exc}); showing MIDI only.")
    return song


def main() -> None:
    st.set_page_config(page_title="MusicGen Music Generator", page_icon="🎵", layout="wide")
    add_style()
    st.session_state.setdefault("songs", [])

    st.markdown(
        '<div class="title-container"><h1 class="title">🎵🎵 Music Generator 🎵🎵</h1></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="sidebar-title-container"><h2 class="sidebar-title">Music Generator</h2></div>',
        unsafe_allow_html=True,
    )

    choice = st.sidebar.radio("Choose a model type:", list(MODELS), horizontal=True)
    model_type = MODELS[choice]
    with st.sidebar.expander("Checkpoint", expanded=False):
        checkpoint = st.text_input("Checkpoint file", str(ROOT / "checkpoints" / model_type / "best.pt"))
        tokenizer = st.text_input("Tokenizer (legacy checkpoints only)", "")

    st.sidebar.markdown('<hr class="hr-style">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header">Start Note For Generator</div>', unsafe_allow_html=True)
    prompt = st.sidebar.text_area(
        " ",
        value="",
        height=200,
        placeholder=PLACEHOLDER,
        help="Start your ABC notation here (empty = let the model start)",
    )

    st.sidebar.markdown('<hr class="hr-style">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header">Adjust Model Settings</div>', unsafe_allow_html=True)
    max_new_tokens = st.sidebar.slider("Max length of generated tokens:", 256, 1024, 512)
    temperature = st.sidebar.slider("Temperature:", 0.4, 1.0, 0.96, step=0.01)
    top_k = st.sidebar.slider("top_k (0 = off):", 0, 10, 0)

    st.sidebar.markdown('<hr class="hr-style">', unsafe_allow_html=True)
    if st.sidebar.button(f"Generate With {choice}", use_container_width=True):
        if not Path(checkpoint).is_file():
            st.error(f"No checkpoint at `{checkpoint}`. Train one first: `musicgen-train --model {model_type}`.")
            return
        spinner = st.empty()
        spinner.markdown(SMILE_SPINNER, unsafe_allow_html=True)
        loaded = get_model(checkpoint, tokenizer)
        abc = generate_abc(loaded.model, loaded.tokenizer, prompt, max_new_tokens, temperature, top_k)
        try:
            if abc is None:
                raise ValueError("empty generation")
            name = f"{model_type}_song_{datetime.now():%Y-%m-%d_%H-%M-%S}"
            st.session_state.songs.insert(0, synthesize(abc, name))
        except Exception:  # noqa: BLE001 - invalid ABC from the model
            spinner.empty()
            st.markdown(ERROR_MESSAGE, unsafe_allow_html=True)
            if abc:
                with st.expander("Show the generated ABC"):
                    st.code(abc, language=None)
            return
        spinner.empty()

    if st.session_state.songs:
        cols = st.columns(2)
        for i, song in enumerate(st.session_state.songs):
            with cols[i % 2]:
                render_song(song, i)
    else:
        st.markdown(
            '<div class="input-section"><p class="input-instructions">'
            "Set a start For ABC annotation and Click Generate</p></div>",
            unsafe_allow_html=True,
        )


main()
