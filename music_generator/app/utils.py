import base64
import shutil
import regex as re
import torch
import os
from datetime import datetime
from music21 import converter
from music_generator.model.MGTransformer import MGTransformer
from music_generator.tokenizing.tokenizer.MGTokenizer import MGTokenizer

def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MGTransformer(**checkpoint['model_args'])
    model.to(device)
    model.eval()

    try:
        model.load_state_dict(checkpoint['model'], strict=True)
    except RuntimeError as e:
        print(f"Failed to load all parameters: {e}")

    optimizer = None
    if 'optimizer_state_dict' in checkpoint:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model

def load_tokenizer(tokenizer_path):
    tokenizer = MGTokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer

def generate_songs(abc_string, abc_dir):
    pattern = re.compile(r'<SOS>(.*?)<EOS>', re.DOTALL)
    matches = pattern.findall(abc_string)
    abc_strings = [match.strip() for match in matches if match.strip()]

    os.makedirs(abc_dir, exist_ok=True)
    is_exist = False
    
    for i, match in enumerate(abc_strings):
        curr_dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            song = converter.parse(match, format='abc')
            midi_path = os.path.join(abc_dir, f'abs_song_{curr_dtime}_{i + 1}.mid')
            song.write('midi', fp=midi_path)
            is_exist = True
        except Exception as e:
            pass
    
    return is_exist

def generate_string(sos_token, model, tokenizer, device, max_new_tokens=512, temperature=1.0, top_k=None):
    in_tokens = tokenizer.encode(sos_token)
    in_tokens_tensor = torch.tensor([in_tokens], dtype=torch.long).to(device)
    out_tokens = model.generate(in_tokens_tensor, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    out_str = tokenizer.decode(out_tokens[0].tolist())
    return out_str


def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')