# Legacy code (reference only)

The original version of MGTLR, kept exactly as it was: the `music_generator/`
package, its Poetry `pyproject.toml` and `poetry.lock`. It is **not
maintained**; use `src/mgtlr/` (see the main README).

Known problems in this version, fixed in `src/mgtlr/`:

- several packages had no `__init__.py`, so imports such as
  `from music_generator.model.TRF import MGTransformer` failed;
- the from-scratch RNN/LSTM created **one cell per time step** (no weight
  sharing) and the torch-API versions stacked `seq_len` layers;
- `rnn_train.py` passed arguments to `MGConfig()`, which takes none, and every
  training script called `save_checkpoints` with a missing argument;
- RNN/LSTM `generate()` passed the hidden state as the targets;
- the Transformer's `generate()` cropped its context to `b_size` (32 tokens)
  instead of `max_seq_len`;
- the tokenizer numbered its special tokens with a gap, so
  `len(tokenizer.vocab)` was one smaller than the largest id;
- the Streamlit model selector was not connected to any model.

Checkpoints and tokenizers trained with this code still load in the new
package (`mgtlr-generate --checkpoint mg_chpts_v1.pth --tokenizer mgt_tokenizer_v1.model`);
a unit test checks this.
