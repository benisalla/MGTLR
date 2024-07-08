from music_generator.tokenizing.tokenizer.KerasTokenizer import KerasTokenizer


load_path = "./music_generator/src/tokenizer/keras_tokenizer_test.json"
tokenizer = KerasTokenizer()
tokenizer.load(load_path)

# Test with some text
sample_texts = ["Hello world!", "Keras is fun for natural language processing."]
sequences = tokenizer.texts_to_sequences(sample_texts)

# Print tokenizer properties and test results
print("Vocabulary Size:", tokenizer.get_vocab_size())
print("Word Index (partial):", dict(list(tokenizer.get_word_index().items())[:10]))  # Print first 10 items for brevity
print("Index to Word Mapping (partial):", dict(list(tokenizer.get_index_word().items())[:10]))  # Print first 10 mappings
print("Sequences for Sample Texts:", sequences)

sequences = tokenizer.texts_to_sequences([
    "Data Science: Data Visualization",
    "Technologies: Python, FastAPI",
    "<EOS>"
])
print("Tokenized sequences:", sequences)

vocab_size = tokenizer.get_vocab_size()
print("Vocabulary size:", vocab_size)

word_index = tokenizer.get_word_index()
print("Word Index:", word_index)

index_word = tokenizer.get_index_word()
print("Index Word:", index_word)
