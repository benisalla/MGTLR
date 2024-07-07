import base64
import streamlit as st
from music_generator.app.utils import clear_directory, generate_songs, generate_string, load_model, load_tokenizer
import torch
import os
from glob import glob
from midi2audio import FluidSynth

def add_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    with open("C:\\Users\\Omar\\Desktop\\Week-end-projects\\music_generator_with_3_nlp_algorithms\\music_generator\\app\\style\\style.css", "r") as css_file:
        css_content = css_file.read()
        css_content = css_content.replace("{encoded_image}", encoded_image)
    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True
    )

sound_font_path = "./music_generator/app/src/luidR3_GM.sf2"

def convert_midi_to_wav(mid_file):
    fs = FluidSynth(sound_font_path)
    wav_file = mid_file.replace('.mid', '.wav')
    fs.midi_to_audio(mid_file, wav_file)
    return wav_file

def display_mid_files(abc_dir):
    mid_files = glob(os.path.join(abc_dir, "*.mid"))
    if not mid_files:
        st.warning("No MIDI files found in the directory.")
    else:
        for mid_file in mid_files:
            file_name = os.path.basename(mid_file)
            st.markdown(f"#### {file_name}")
            wav_file = convert_midi_to_wav(mid_file)
            audio_bytes = open(wav_file, 'rb').read()
            st.audio(audio_bytes, format='audio/wav')


def main():
    
    bk_img = "./music_generator/app/src/image.png"
    tokenizer_path = "./music_generator/src/tokenizer/mgt_tokenizer_v1.model"  
    checkpoint_path = './music_generator/src/checkpoints/mg_chpts_v1.pth' 
    abc_dir = "./music_generator/app/abs_dir"
    
    clear_directory(abc_dir)
    add_background_image(bk_img)
    
    st.markdown('<div class="title-container"><h1 class="title">🎵 Music Generator App</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-container"><p class="instructions">Input a starting sequence, and the model will generate music for you.</p></div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="sidebar-title">Music Generator</div>', unsafe_allow_html=True)
    
    abc_dir = "generated_songs"
    os.makedirs(abc_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer(tokenizer_path)

    st.sidebar.header("Input Music Sequence")
    start_sequence = st.sidebar.text_area("Enter the starting sequence of notes (e.g., <SOS>X:1...)", value="<SOS>X:1\n")

    max_length = st.sidebar.slider("Max length of generated music", min_value=256, max_value=1024, value=512)
    start_it = st.sidebar.text_input("Enter the start of your music sequence", "")

    if st.sidebar.button("Generate Music"):
        if start_sequence:
            in_str = f"<SOS>{start_it}"
            abc_string = generate_string(in_str, model, tokenizer, device, max_new_tokens=max_length, temperature=1.0)
            
            if not isinstance(abc_string, str):
                st.error("Generated string is not valid. Please try again.")
                return
            
            is_songs_exist = generate_songs(abc_string, abc_dir)
            
            if is_songs_exist:
                st.markdown('<div class="generated-container">', unsafe_allow_html=True)
                st.text_area("Generated Music", value=abc_string, height=400)
                st.markdown('</div>', unsafe_allow_html=True)
                display_mid_files(abc_dir)
            else:
                st.error("The generated songs are not clean. Please try again.")
        else:
            st.markdown('<div class="upload-container"><p class="instructions">Please input a starting sequence to generate music.</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# poetry run streamlit run music_generator/app/main.py