# Training guide

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'           # core + tests + music21 + matplotlib
# GPU: install the CUDA build of torch first, see https://pytorch.org/get-started
# CPU only: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 1. Data

Nothing to prepare by hand. On first use, `train.json` (80 MB) and
`validation.json` from [IrishMAN](https://huggingface.co/datasets/sander-wood/irishman)
are downloaded into `data/`. The tokenized corpus is cached in `data/cache/`
(first encoding ~1.5 min, then instant).

Your own tunes: any JSON list of `{"abc notation": "X:1\n..."}` records works:

```bash
musicgen-train --train-file my_train.json --val-file my_val.json
```

## 2. Tokenizer

```bash
musicgen-tokenizer --vocab-size 856 --sample-tunes 20000    # ~1 min on CPU
```

Writes `tokenizer/mgt_tokenizer.model` (+ a readable `.vocab`). If you skip
this step, `musicgen-train` trains the same tokenizer automatically. All models
that should be compared must use the same tokenizer.

## 3. Train

```bash
musicgen-train --model transformer      # -> checkpoints/transformer/
musicgen-train --model lstm             # -> checkpoints/lstm/
musicgen-train --model rnn              # -> checkpoints/rnn/
musicgen-train --model lstm --impl torch   # cuDNN LSTM, much faster than the from-scratch cells
```

Each preset comes from the original project (`config.py`). Any flag overrides it:

| flag | transformer | lstm / rnn | meaning |
|------|------------:|-----------:|---------|
| `--d-model` | 128 | 128 | token embedding size |
| `--layers` | 5 | 2 | blocks / stacked recurrent layers |
| `--head-dim` | 32 | - | attention head size (4 heads) |
| `--max-seq-len` | 1024 | - | context length |
| `--hidden-dim` | - | 256 | recurrent state size |
| `--impl` | - | scratch | `scratch` or `torch` |
| `--activation` | - | tanh | RNN only: `tanh` or `relu` |
| `--seq-len` | 512 | 200 | tokens per training window |
| `--lr` | 2e-4 | 1e-3 | peak learning rate |
| `--weight-decay` | 1e-2 | 1e-3 | AdamW decay |
| `--batch-size` | 32 | 32 | windows per step |
| `--max-steps` | 20000 | 20000 | optimizer steps |
| `--warmup-steps` | 500 | 500 | linear warmup before cosine decay |

Other flags: `--grad-accum-steps`, `--grad-clip`, `--label-smoothing`,
`--dropout`, `--precision {auto,fp32,bf16,fp16}`, `--device`,
`--eval-every`, `--eval-batches`, `--log-every`, `--seed`,
`--train-limit N` (first N tunes only). See `musicgen-train --help`.

### Output

```
checkpoints/transformer/
  best.pt               lowest validation loss  (use this one)
  last.pt               latest step             (resume from this one)
  history.json          train / val loss, accuracy, perplexity, lr
  training_curves.png   loss and accuracy plots
```

### Resume

```bash
musicgen-train --model transformer --resume checkpoints/transformer/last.pt --max-steps 40000
```

Weights, optimizer state, step count, best loss and history continue. Keep
the same model flags as the first run.

### Hardware

Every model trains on CPU, but slowly. On a GPU the Transformer and the
`--impl torch` recurrent models are fast. The `scratch` cells loop over time
steps in Python, so they are the slowest; use them to learn how the cells
work, or for short runs.

## Results of the original training runs

1. **Transformer (TRF)**

   ![Transformer Training Results](https://github.com/user-attachments/assets/bcf83ee8-a9ef-4cdb-83d2-435f07a00f49)

2. **Recurrent Neural Network (RNN)**

   ![RNN Training Results](https://github.com/user-attachments/assets/49a4e22b-5851-4817-91de-311b398c330d)

3. **Long Short-Term Memory (LSTM)**

   ![LSTM Training Results](https://github.com/user-attachments/assets/72b4d3fb-3cf4-444e-bbc4-02695c3c117c)
