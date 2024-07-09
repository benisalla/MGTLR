import base64
import json
import shutil
import regex as re
import torch
import os
from datetime import datetime
from midi2audio import FluidSynth
from pydub import AudioSegment
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

def generate_songs(abc_string, abc_dir, json_file_path):
    pattern = re.compile(r'<SOS>(.*?)<EOS>', re.DOTALL)
    matches = pattern.findall(abc_string)
    abc_strings = [match.strip() for match in matches if match.strip()]

    os.makedirs(abc_dir, exist_ok=True)
    is_exist = False
    song_details = load_song_details(json_file_path)
    
    for i, match in enumerate(abc_strings):
        curr_dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            song = converter.parse(match, format='abc')
            midi_file_name = f'abs_song_{curr_dtime}_{i + 1}.mid'
            midi_path = os.path.join(abc_dir, midi_file_name)
            song.write('midi', fp=midi_path)
            is_exist = True

            wav_file = convert_midi_to_wav(midi_path)
            mp3_file = convert_wav_to_mp3(wav_file)
            os.remove(midi_path)  

            song_details[midi_file_name] = {
                "abc": match,
                "mp3_path": mp3_file
            }

        except Exception as e:
            pass

    save_song_details(song_details, json_file_path)
    return is_exist

def generate_string(sos_token, model, tokenizer, device, max_new_tokens=512, temperature=1.0, top_k=None):
    if top_k == 0:
        top_k = None
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

def save_abc_file(abc_content, file_path):
    with open(file_path, 'w') as abc_file:
        abc_file.write(abc_content)

def convert_midi_to_wav(mid_file):
    fs = FluidSynth()
    wav_file = mid_file.replace('.mid', '.wav')
    fs.midi_to_audio(mid_file, wav_file)
    return wav_file

def convert_wav_to_mp3(wav_file):
    mp3_file = wav_file.replace('.wav', '.mp3')
    audio = AudioSegment.from_wav(wav_file)
    audio.export(mp3_file, format='mp3')
    os.remove(wav_file)
    return mp3_file

def save_song_details(song_details, json_file_path):
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    with open(json_file_path, 'w') as json_file:
        json.dump(song_details, json_file, indent=4)

def load_song_details(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)
    return {}

def add_background_image(image_path, st):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    with open("./music_generator/app/style/style.css", "r") as css_file:
        css_content = css_file.read()
        css_content = css_content.replace("{encoded_image}", encoded_image)
    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True
    )


def display_songs(song_details, st):
    cols = st.columns(2)  
    for i, (file_name, details) in enumerate(song_details.items()):
        col = cols[i % 2]  
        with col:
            with st.container(border=True):                
                audio_bytes = open(details['mp3_path'], 'rb').read()
                st.audio(audio_bytes, format='audio/mp3')
                down_f_name = ".".join(file_name.split(".")[:-1]) + ".mp3"
                st.markdown(f'<a href="{details["mp3_path"]}" download="{down_f_name}" class="download-link">Download MP3 ⬇️</a>', unsafe_allow_html=True)
                
                clear_abc_en = details["abc"].replace('\n', '<br>').encode("utf-8")
                clear_abc_de = clear_abc_en.decode('utf-8', errors='ignore')
                with st.expander("Show ABC Annotations", expanded=False):
                    st.markdown(f'<div class="expander-text">{clear_abc_de}</div>', unsafe_allow_html=True)
                    
def init_session_state(st):
    if 'device' not in st.session_state:
        st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True

    if 'model' not in st.session_state:
        checkpoint_path = './music_generator/src/checkpoints/mg_chpts_v1.pth'
        st.session_state.model = load_model(checkpoint_path, st.session_state.device)

    if 'tokenizer' not in st.session_state:
        tokenizer_path = "./music_generator/src/tokenizer/mgt_tokenizer_v1.model"
        st.session_state.tokenizer = load_tokenizer(tokenizer_path)

    if 'abc_dir' not in st.session_state:
        st.session_state.abc_dir = "./music_generator/app/abc_dir"
        os.makedirs(st.session_state.abc_dir, exist_ok=True)

    if 'json_file_path' not in st.session_state:
        st.session_state.json_file_path = os.path.join(st.session_state.abc_dir, 'song_details.json')

    # if 'start_it' not in st.session_state:
    #     st.session_state.start_it = "X:1\n"

    # if 'max_length' not in st.session_state:
    #     st.session_state.max_length = 512

    # if 'temperature' not in st.session_state:
    #     st.session_state.temperature = 1.0

    # if 'top_k' not in st.session_state:
    #     st.session_state.top_k = 0