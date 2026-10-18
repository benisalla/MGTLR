"""Generated text -> ABC tunes -> MIDI -> audio.

Only ``extract_tunes`` is needed by the core package. The conversions use
optional dependencies (``pip install -e '.[music]'``):

* ``music21``     ABC -> MIDI
* ``fluidsynth``  MIDI -> WAV, a system program (apt/brew) plus a SoundFont (.sf2)
* ``pydub``       WAV -> MP3, needs ``ffmpeg``
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import regex as re

from musicgen.tokenizer import EOS, PAD, SOS

# a tune starts at <SOS> and ends at <EOS>, at the next <SOS> or at the end of the text
_TUNE = re.compile(
    re.escape(SOS) + r"(.*?)(" + re.escape(EOS) + r"|(?=" + re.escape(SOS) + r")|$)", re.DOTALL
)

# Where a General MIDI SoundFont usually lives (Debian/Ubuntu fluid-soundfont-gm, macOS brew)
SOUNDFONT_CANDIDATES = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/default.sf2",
    "/opt/homebrew/share/soundfonts/default.sf2",
)


def extract_tunes(text: str, keep_unfinished: bool = False) -> list[str]:
    """Return the ABC tunes found in generated text.

    A tune closed by <EOS> is finished. One cut off by the token budget or by
    a new <SOS> (no <EOS>) is kept only if ``keep_unfinished`` is True.
    """
    tunes = []
    for match in _TUNE.finditer(text):
        finished = match.group(2) == EOS
        body = match.group(1).replace(PAD, "").strip()
        if body and (finished or keep_unfinished):
            tunes.append(body)
    return tunes


def abc_to_midi(abc: str, midi_path: str | Path) -> Path:
    """Parse ABC with music21 and write a .mid file. Raises if the ABC is invalid."""
    from music21 import converter  # optional dependency

    midi_path = Path(midi_path)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    score = converter.parse(abc, format="abc")
    score.write("midi", fp=str(midi_path))
    return midi_path


def find_soundfont(explicit: str | Path | None = None) -> Path | None:
    for candidate in (explicit, os.environ.get("MUSICGEN_SOUNDFONT"), *SOUNDFONT_CANDIDATES):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def midi_to_wav(midi_path: str | Path, soundfont: str | Path | None = None) -> Path:
    """Render MIDI to WAV with FluidSynth (needs the fluidsynth binary + a SoundFont)."""
    sf2 = find_soundfont(soundfont)
    if sf2 is None:
        raise FileNotFoundError("no SoundFont found: pass one or set MUSICGEN_SOUNDFONT")
    if shutil.which("fluidsynth") is None:
        raise FileNotFoundError("the fluidsynth program is not installed")
    wav_path = Path(midi_path).with_suffix(".wav")
    subprocess.run(
        ["fluidsynth", "-ni", "-F", str(wav_path), "-r", "44100", str(sf2), str(midi_path)],
        check=True,
        capture_output=True,
    )
    return wav_path


def wav_to_mp3(wav_path: str | Path, remove_wav: bool = True) -> Path:
    from pydub import AudioSegment  # optional dependency

    mp3_path = Path(wav_path).with_suffix(".mp3")
    AudioSegment.from_wav(str(wav_path)).export(str(mp3_path), format="mp3")
    if remove_wav:
        Path(wav_path).unlink()
    return mp3_path
