from music_generator.tokenizing import NaiveTokenizer


text = """
Science des Données: Data Manipulation and Cleaning, Data Visualization, Statistical Analysis + EDA, Feature engineering. 
Built a RAG system for Moroccan law documents. Technologies: Llama-Index, FastAPI, 
React/Next.js, TypeScript, Python. Computer science engineering at ENSA, Fes, Morocco from 2021 to now.
"""

tokenizer = NaiveTokenizer(text, eost="%")
print(f"Size of the vocabulary: {tokenizer.vocab_size}")
print("Token to index mapping keys:", tokenizer.itos.keys())

sample_text = "Data Visualization and Machine Learning."
encoded_sample = tokenizer.encode(sample_text)
decoded_sample = tokenizer.decode(encoded_sample)
print("Encoded Sample:", encoded_sample)
print("Decoded Sample:", decoded_sample)