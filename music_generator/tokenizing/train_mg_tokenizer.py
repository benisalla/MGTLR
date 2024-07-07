# Initialize DataLoader
from music_generator.data.dataloader import DataLoader
from music_generator.tokenizing.tokenizer.MGTokenizer import MGTokenizer

main_dir = "./music_generator/src"

train_path = f'{main_dir}/dataset/train_abc.json'
val_path = f'{main_dir}/dataset/val_abc.json'
save_path = f"{main_dir}/tokenizer/mgt_tokenizer_test"

print("loading the data ...")
data_loader = DataLoader(train_path, val_path)

# Load and preprocess data
train_text, val_text = data_loader.get_data()
print(f"train text: {train_text[:10]}")
print(f"val text: {val_text[:10]}")

# Tokenizer training
tokenizer = MGTokenizer()
vocab_size = 856

print("training the tokenizer just started ...")
tokenizer.train(val_text, vocab_size=vocab_size, verbose=True)

# Update and register special tokens
special_tokens = {
    '<PAD>': vocab_size + 1,
    '<EOS>': vocab_size + 2,
    '<SOS>': vocab_size + 3,
    '<OOV>': vocab_size + 4,
}

# Update the vocab_size to reflect new tokens
vocab_size += len(special_tokens)  
tokenizer.register_special_tokens(special_tokens)

# Display and save the tokenizer
print("Vocab Sample:", list(tokenizer.vocab.items())[-10:])
tokenizer.save(f"../src/dataset/mgt_tokenizer_test")