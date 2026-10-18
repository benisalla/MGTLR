"""Small CPU tests for the three architectures (and both RNN/LSTM implementations)."""

import pytest
import torch

from musicgen.config import make_model_config
from musicgen.models import MGLSTM, MGRNN, MGTransformer, build_model, sample_next

VOCAB = 40
VARIANTS = [
    ("transformer", "scratch"),
    ("lstm", "scratch"),
    ("lstm", "torch"),
    ("rnn", "scratch"),
    ("rnn", "torch"),
]
TINY = {"d_model": 16, "head_dim": 8, "n_layers": 2, "hidden_dim": 24, "max_seq_len": 32}


def make(model_type: str, impl: str = "scratch", seed: int = 0, **overrides):
    torch.manual_seed(seed)
    return build_model(make_model_config(model_type, VOCAB, impl=impl, **{**TINY, **overrides}))


@pytest.mark.parametrize(("model_type", "impl"), VARIANTS)
def test_shapes_and_loss(model_type, impl):
    model = make(model_type, impl)
    x = torch.randint(0, VOCAB, (3, 10))
    out = model(x, x)
    assert out.logits.shape == (3, 10, VOCAB)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    # random init should be close to uniform guessing: ln(40) = 3.69
    assert abs(out.loss.item() - torch.log(torch.tensor(float(VOCAB))).item()) < 0.5
    assert model(x).loss is None


def test_factory_classes():
    assert isinstance(make("transformer"), MGTransformer)
    assert isinstance(make("lstm"), MGLSTM)
    assert isinstance(make("rnn"), MGRNN)


@pytest.mark.parametrize(("model_type", "impl"), VARIANTS)
def test_causality(model_type, impl):
    """Changing a future token must not change the logits of earlier positions."""
    model = make(model_type, impl).eval()
    x = torch.randint(0, VOCAB, (2, 12))
    y = x.clone()
    y[:, 7:] = (y[:, 7:] + 1) % VOCAB
    with torch.no_grad():
        a, b = model(x).logits, model(y).logits
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5)
    assert not torch.allclose(a[:, 7:], b[:, 7:])


@pytest.mark.parametrize(("model_type", "impl"), [v for v in VARIANTS if v[0] != "transformer"])
def test_state_carries_context(model_type, impl):
    """Feeding a sequence in two halves with the state == feeding it at once."""
    model = make(model_type, impl).eval()
    x = torch.randint(0, VOCAB, (2, 10))
    with torch.no_grad():
        full = model(x).logits
        first = model(x[:, :6])
        second = model(x[:, 6:], state=first.state)
    assert torch.allclose(torch.cat([first.logits, second.logits], 1), full, atol=1e-5)


@pytest.mark.parametrize("kind", ["rnn", "lstm"])
@pytest.mark.parametrize("bias", [False, True])
def test_scratch_matches_torch(kind, bias):
    """The hand-written cells compute exactly what nn.RNN / nn.LSTM compute."""
    scratch = make(kind, "scratch", bias=bias)
    reference = make(kind, "torch", bias=bias, seed=1)
    with torch.no_grad():
        reference.emb.weight.copy_(scratch.emb.weight)
        reference.fc.load_state_dict(scratch.fc.state_dict())
        for layer, cell in enumerate(scratch.rnn.cells):
            getattr(reference.rnn, f"weight_ih_l{layer}").copy_(cell.i2h.weight)
            getattr(reference.rnn, f"weight_hh_l{layer}").copy_(cell.h2h.weight)
            if bias:
                getattr(reference.rnn, f"bias_ih_l{layer}").copy_(cell.i2h.bias)
                getattr(reference.rnn, f"bias_hh_l{layer}").copy_(cell.h2h.bias)
    x = torch.randint(0, VOCAB, (3, 9))
    with torch.no_grad():
        assert torch.allclose(scratch(x).logits, reference(x).logits, atol=1e-5)


def test_recurrent_weights_are_shared_over_time():
    """One cell per layer, whatever the sequence length (the legacy code had one per step)."""
    model = make("lstm", n_layers=2)
    assert len(model.rnn.cells) == 2
    x = torch.randint(0, VOCAB, (1, 25))
    assert model(x).logits.shape == (1, 25, VOCAB)


def test_transformer_weight_tying_and_max_len():
    model = make("transformer")
    assert model.lm_head.weight is model.decoder.tok_emb.weight
    with pytest.raises(ValueError):
        model(torch.zeros(1, 33, dtype=torch.long))


@pytest.mark.parametrize(("model_type", "impl"), VARIANTS)
def test_generate(model_type, impl):
    model = make(model_type, impl)
    prompt = torch.randint(0, VOCAB, (2, 3))
    out = model.generate(prompt, max_new_tokens=40, temperature=1.0, top_k=5)
    assert out.shape == (2, 43)  # also checks the transformer crops to max_seq_len=32
    assert torch.equal(out[:, :3], prompt)
    assert model.training  # generate() restores the mode


def test_generate_stops_at_eos():
    model = make("transformer")
    with torch.no_grad():  # make token 7 overwhelmingly likely
        model.lm_head.weight.zero_()
        model.lm_head.weight[7] = 1.0
        model.decoder.f_ln.w.fill_(5.0)
    out = model.generate(torch.zeros(1, 2, dtype=torch.long), 20, temperature=0, eos_id=7)
    assert out.shape == (1, 3) and out[0, -1] == 7


def test_sampling():
    logits = torch.tensor([[0.0, 5.0, 1.0, 4.0]])
    assert sample_next(logits, temperature=0).item() == 1
    g = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, top_k=2, generator=g).item() for _ in range(50)}
    assert picks <= {1, 3}


@pytest.mark.parametrize(("model_type", "impl"), VARIANTS)
def test_can_overfit_one_batch(model_type, impl):
    """If a model cannot memorise one batch, the forward/backward pass is broken."""
    model = make(model_type, impl)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, VOCAB, (4, 12))
    y = torch.roll(x, -1, dims=1)
    first = model(x, y).loss.item()
    for _ in range(150):
        loss = model(x, y).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.2 * first
