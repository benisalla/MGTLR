"""The three music language models and a factory to build one from its config."""

from musicgen.config import ModelConfig
from musicgen.models.base import IGNORE_INDEX, LanguageModel, ModelOutput, sample_next
from musicgen.models.recurrent import MGLSTM, MGRNN
from musicgen.models.transformer import MGTransformer

MODEL_CLASSES: dict[str, type[LanguageModel]] = {
    "transformer": MGTransformer,
    "lstm": MGLSTM,
    "rnn": MGRNN,
}


def build_model(config: ModelConfig) -> LanguageModel:
    return MODEL_CLASSES[config.model_type](config)


__all__ = [
    "IGNORE_INDEX",
    "MGLSTM",
    "MGRNN",
    "MODEL_CLASSES",
    "LanguageModel",
    "MGTransformer",
    "ModelOutput",
    "build_model",
    "sample_next",
]
