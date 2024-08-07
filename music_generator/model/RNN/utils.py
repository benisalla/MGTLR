import random
import regex as re
import torch
import os
from datetime import datetime
from music21 import converter

def generate_songs(abc_string: str, abc_dir: str) -> bool:
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
            print(f"Error processing song {i + 1}: {e}")

    return is_exist

def generate_string(sos_token, model, tokenizer, device: torch.device, max_new_tokens = 512, temperature = 1.0, top_k = None) -> str:

    out_str = model.generate(tokenizer, sos_token, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    pattern = re.compile(r'<SOS>(.*?)<EOS>', re.DOTALL)
    matches = pattern.findall(out_str)
    
    return ''.join(matches) if matches else ''