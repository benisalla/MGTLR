# Testing guide

Three levels of checks, from seconds to minutes:

1. **Unit tests** (`pytest`): every module, CPU only, no network.
2. **Overfit sanity check**: real data, can each model memorise 8 tunes?
3. **End-to-end checks**: the real CLIs and the app.

## 1. Unit tests

```bash
pytest                         # everything (~1 min on a laptop CPU)
pytest tests/test_models.py    # one file
pytest -k lstm -v              # tests whose name contains "lstm"
ruff check src tests app       # lint
ruff format --check src tests app
```

No GPU and no internet needed: the tests use four short ABC tunes, a tiny BPE
tokenizer and tiny models.

| File | What it proves |
|------|----------------|
| `test_config.py` | presets + overrides; invalid settings (unknown model, bad head size, bad precision...) are rejected early |
| `test_tokenizer.py` | encode/decode round trip on seen **and unseen** text; special tokens are single ids; save/load and dict round trips; **tokenizer files from the legacy code load**, with the id gap handled |
| `test_data.py` | tunes are framed `<SOS>…<EOS>`; batches are contiguous windows with targets shifted by one; seeded batches are reproducible; the token cache; `--overfit` uses the same tunes for train and validation |
| `test_models.py` | for all 5 variants (Transformer, LSTM and RNN x scratch/torch): output shapes, loss ≈ ln(V) at init, **causality** (future tokens cannot change past logits), `generate()` length / prompt / mode restore, and each variant **memorises one batch**; recurrent state carries context across calls; **the hand-written RNN/LSTM cells equal `nn.RNN` / `nn.LSTM`** (weights copied across, with and without bias); weights are shared across time steps; weight tying; early stop at `<EOS>`; greedy and top-k sampling |
| `test_train.py` | warmup + cosine LR values; optimizer groups (no decay on norms/biases, tied weight counted once); a short run of **each model** lowers the loss and writes `last.pt` / `best.pt` / `history.json` / curves; the reloaded checkpoint gives identical logits; **resume** continues the step count; gradient accumulation; the CLI end to end, including auto-training a missing tokenizer |
| `test_generate.py` | checkpoint round trip; **a checkpoint saved by the legacy `MGTransformer` loads and gives the same logits**; tune extraction from generated text; seeded generation is reproducible; the `musicgen-generate` CLI writes clean `.abc` files; ABC -> MIDI |
| `test_app.py` | the Streamlit app runs headless; a missing checkpoint and invalid generated ABC show messages instead of crashing (skipped if streamlit is not installed) |

## 2. Overfit sanity check (real data)

The most useful test of a training pipeline: if a model cannot memorise a
handful of tunes, something is broken (targets not shifted, mask wrong, LR far
off...). The default model sizes are used; only the windows are shortened,
because the 8 tunes are just 1,055 tokens.

```bash
for m in transformer lstm rnn; do
  musicgen-train --model $m --overfit 8 --max-steps 300 --batch-size 8 --seq-len 128 \
      --lr 2e-3 --warmup-steps 20 --eval-every 100 --log-every 25 --eval-batches 5 \
      --output-dir checkpoints/overfit-$m
done
```

Results on a shared laptop CPU (validation = the same 8 tunes; random
guessing starts at ln 859 = 6.76):

| model | params | val loss @100 | @200 | @300 | val accuracy @300 | time |
|-------|-------:|-----:|-----:|-----:|-----:|-----:|
| RNN (scratch) | 0.56M | 0.207 | 0.070 | **0.058** | 99.0% | 3.5 min |
| Transformer | 1.23M | 1.020 | 0.379 | **0.248** | 92.5% | 3.5 min |
| LSTM (scratch) | 1.25M | 3.596 | 2.787 | **2.600** | 37.0% | 7.5 min |

The LSTM learns more slowly than the other two on this task: 300 steps are
not enough. Given more steps it memorises the tunes too. The run below uses
`--impl torch`, which computes the same thing as the scratch cells (see
`test_scratch_matches_torch`) but faster; lr 3e-3, 1,000 steps:

| model | val loss @250 | @500 | @750 | @1000 |
|-------|-----:|-----:|-----:|-----:|
| LSTM (torch) | 1.371 | 0.211 | 0.093 | **0.079** (98.9% accuracy) |

Generating from the overfit RNN at temperature 0.5 gives back a complete
training jig and stops on its own at `<EOS>`. music21 accepts it and writes
a MIDI file:

```
X:1
L:1/8
M:6/8
K:G
 (G/A/B).D D>ED | DcB AGA | (G/A/B).D D>ED | GAG GED | (G/A/B).D DED | DcB cde | dBG ABc |
 BcA GED :: GBd GBd | GBd ecA | GBd efg | GAG GED | GBd GBd | GBd def | gfg dBA | GAG GED :|
```

What this does **not** tell you: whether a model generalises to new tunes.
That needs a full training run (GPU recommended) and listening to the results.

## 3. End-to-end checks

```bash
musicgen-tokenizer --sample-tunes 20000                       # ~1 min, 859-token vocabulary
musicgen-generate --checkpoint checkpoints/overfit-rnn/best.pt --midi --temperature 0.5
streamlit run app/main.py                                  # pick RNN, set the checkpoint path, Generate
```

Checked during development: the full IrishMAN training split (214,122 tunes)
encodes to 35.8M tokens in ~1.5 min (cached afterwards). The app generated the
tune above and offered it as MIDI. MP3 playback additionally needs `fluidsynth`
and `ffmpeg` (see [inference.md](inference.md)).
