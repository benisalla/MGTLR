# Generating music

## Command line

```bash
musicgen-generate --checkpoint checkpoints/transformer/best.pt --num-tunes 3 --midi
```

Each tune is printed and saved as `outputs/tune_01.abc`, `outputs/tune_01.mid`, ...

| flag | default | meaning |
|------|---------|---------|
| `--prompt` | `X:1\nL:1/8\nM:6/8\nK:G\n` | start of the tune in ABC (`\n` allowed) |
| `--prompt-file` | - | read the prompt from a file |
| `--max-new-tokens` | 512 | token budget; generation also stops at `<EOS>` |
| `--temperature` | 0.9 | lower = safer, higher = more surprising; 0 = greedy |
| `--top-k` | 0 | sample among the k most likely tokens only (0 = off) |
| `--seed` | - | make sampling reproducible |
| `--midi` | off | also write `.mid` files (needs `music21`) |
| `--tokenizer` | - | only for checkpoints made with the legacy code |

The prompt sets the style. The header fields matter most: `M:` metre (6/8 jig,
4/4 reel, 3/4 waltz), `K:` key, `L:` default note length.

```bash
musicgen-generate --checkpoint checkpoints/lstm/best.pt \
    --prompt 'X:1\nL:1/8\nM:3/4\nK:Amin\nA,B,CE |' --temperature 0.8 --seed 0
```

Checkpoints from the original project load too:

```bash
musicgen-generate --checkpoint mg_chpts_v1.pth --tokenizer mgt_tokenizer_v1.model
```

## Python

```python
from musicgen.checkpoint import load_checkpoint
from musicgen.generate import generate_abc
from musicgen.music import abc_to_midi

loaded = load_checkpoint("checkpoints/transformer/best.pt")
abc = generate_abc(loaded.model, loaded.tokenizer, "X:1\nM:6/8\nK:D\n", temperature=0.9, seed=0)
abc_to_midi(abc, "tune.mid")
```

## Streamlit app

<div align="center">
  <img src="https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/a3eed552-ff00-4cd0-920a-9bd02175ba9e" width="600" height="300"/>
</div>

```bash
pip install -e '.[app]'
streamlit run app/main.py
```

Choose TRF, LSTM or RNN, write the start of a tune (or leave it empty) and
press **Generate**. The app loads `checkpoints/<model>/best.pt` (the path can
be changed in the sidebar), then shows the ABC, a MIDI download and, when
audio is available, a player and an MP3 download.

Audio needs two system tools: `fluidsynth` with a General MIDI SoundFont,
and `ffmpeg` for MP3.

```bash
sudo apt install fluidsynth fluid-soundfont-gm ffmpeg     # Debian / Ubuntu
brew install fluid-synth ffmpeg                          # macOS (+ any .sf2 file)
export MUSICGEN_SOUNDFONT=/path/to/FluidR3_GM.sf2           # if not found automatically
```

Without them, the app still shows the ABC and the MIDI file.

Some generated tunes are not valid ABC (a model can break a bar or forget the
key). The app then shows the error card and the raw text. Try again with
another prompt or a lower temperature.
