from music_generator.tokenizing import KerasTokenizer


texts = [
    "Data Science: Data Manipulation and Cleaning, Data Visualization, Statistical Analysis + EDA, Feature engineering.",
    "Built a RAG system for Moroccan law documents. Technologies: Llama-Index, FastAPI, React/Next.js, TypeScript, Python.",
    "Computer science engineering at ENSA, Fes, Morocco from 2021 to now. Integrated Preparatory Classes from 2019 to 2021."
]
tokenizer = KerasTokenizer()
tokenizer.fit(texts)

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