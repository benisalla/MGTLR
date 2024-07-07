from typing import List, Dict
from tensorflow.keras.preprocessing.text import Tokenizer


class KerasTokenizer:
    def __init__(
        self,
        num_words: int = None,
        filters: str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower: bool = True,
        split: str = " ",
        char_level: bool = False,
        oov_token: str = "<OOV>",
    ) -> None:
        """Initialize the tokenizer with specified configurations"""
        self.tokenizer = Tokenizer(
            num_words=num_words,
            filters=filters,
            lower=lower,
            split=split,
            char_level=char_level,
            oov_token=oov_token,
        )
        self.vocab_size = 0

    def fit(self, texts: List[str]) -> None:
        """Fit the tokenizer on the provided texts and update special tokens"""
        self.tokenizer.fit_on_texts(texts)
        self._update_special_tokens()

    def _update_special_tokens(self) -> None:
        """Private method to add special tokens to the tokenizer after fitting on initial texts."""
        max_index = max(self.tokenizer.word_index.values(), default=0)
        special_tokens = {
            "<PAD>": max_index + 1,
            "<SOS>": max_index + 2,
            "<EOS>": max_index + 3,
        }
        self.tokenizer.word_index.update(special_tokens)
        self.tokenizer.index_word = {v: k for k, v in self.tokenizer.word_index.items()}
        self.vocab_size = len(self.tokenizer.word_index)

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts to sequences of integers"""
        return self.tokenizer.texts_to_sequences(texts)

    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary"""
        return self.vocab_size

    def get_word_index(self) -> Dict[str, int]:
        """Return the word index dictionary"""
        return self.tokenizer.word_index

    def get_index_word(self) -> Dict[int, str]:
        """Return the index to word mapping"""
        return self.tokenizer.index_word
