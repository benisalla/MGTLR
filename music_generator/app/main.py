import base64
import streamlit as st
from music_generator.app.utils import clear_directory, generate_songs, generate_string, load_model, load_song_details, load_tokenizer, save_abc_file
import torch
import os
from datetime import datetime

def add_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    with open("./music_generator/app/style/style.css", "r") as css_file:
        css_content = css_file.read()
        css_content = css_content.replace("{encoded_image}", encoded_image)
    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True
    )


def display_songs(song_details):
    cols = st.columns(2)  
    for i, (file_name, details) in enumerate(song_details.items()):
        col = cols[i % 2]  
        with col:
            with st.container(border=True):                
                audio_bytes = open(details['mp3_path'], 'rb').read()
                st.audio(audio_bytes, format='audio/mp3')
                down_f_name = ".".join(file_name.split(".")[:-1]) + ".mp3"
                st.markdown(f'<a href="{details["mp3_path"]}" download="{down_f_name}" class="download-link">Download MP3 ⬇️</a>', unsafe_allow_html=True)
                
                clear_abc = details["abc"].replace('\n', '<br>')
                with st.expander("Show ABC Annotations", expanded=False):
                    st.markdown(f'<div class="expander-text">{clear_abc}</div>', unsafe_allow_html=True)

def main():
    bk_img = "./music_generator/app/src/music-bkgd.jpeg" 
    tokenizer_path = "./music_generator/src/tokenizer/mgt_tokenizer_v1.model"  
    checkpoint_path = './music_generator/src/checkpoints/mg_chpts_v1.pth' 
    abc_dir = "./music_generator/app/abc_dir"
    json_file_path = os.path.join(abc_dir, 'song_details.json')
    
    os.makedirs(abc_dir, exist_ok=True)
    clear_directory(abc_dir)
    add_background_image(bk_img)
    
    st.markdown('<div class="title-container"><h1 class="title">🎵🎵 Music Generator 🎵🎵</h1></div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="sidebar-title">Music Generator</div>', unsafe_allow_html=True)
    
    os.makedirs(abc_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer(tokenizer_path)

    st.sidebar.header("Input Music Sequence")
    start_it = st.sidebar.text_area("Enter the start of your music sequence (e.g., X:1...)", value="X:1\n")

    max_length = st.sidebar.slider("Max length of generated music", min_value=256, max_value=1024, value=512)

    if st.sidebar.button("Generate Music"):
        if start_it:
            in_str = f"<SOS>{start_it}"
            abc_string = generate_string(in_str, model, tokenizer, device, max_new_tokens=max_length, temperature=1.0)
            
            if not isinstance(abc_string, str):
                st.error("Generated string is not valid. Please try again.")
                return
            
            curr_dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            abc_file_path = os.path.join(abc_dir, f'abs_song_{curr_dtime}.abc')
            save_abc_file(abc_string, abc_file_path)
            
            is_songs_exist = generate_songs(abc_string, abc_dir, json_file_path)
            
            if is_songs_exist:                
                song_details = load_song_details(json_file_path)
                display_songs(song_details)
            else:
                error_message = """
                <div class="error-component">
                    <strong>Error:</strong> Unable to generate valid songs from the provided input. Please check your ABC notation for errors, adjust it if necessary, and try again. Ensure that the notation follows the correct format to allow successful song generation.
                </div>
                """
                st.markdown(error_message, unsafe_allow_html=True)

    else:
        st.markdown('''
        <div class="input-section">
            <p class="input-instructions">Set a start For ABC annotation and Click Generate</p>
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()