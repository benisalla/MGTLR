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
            st.markdown(f"### {file_name}")
            audio_bytes = open(details['mp3_path'], 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')
            st.markdown(f"[Download {file_name}]({details['mp3_path']})")
            with st.expander("Show ABC Annotations"):
                st.text(details['abc'])

def main():
    bk_img = "./music_generator/app/src/music-bkgd.jpeg" 
    tokenizer_path = "./music_generator/src/tokenizer/mgt_tokenizer_v1.model"  
    checkpoint_path = './music_generator/src/checkpoints/mg_chpts_v1.pth' 
    abc_dir = "./music_generator/app/abc_dir"
    json_file_path = os.path.join(abc_dir, 'song_details.json')
    
    os.makedirs(abc_dir, exist_ok=True)
    clear_directory(abc_dir)
    add_background_image(bk_img)
    
    st.markdown('<div class="title-container"><h1 class="title">🎵🎵 Music Generator App 🎵🎵</h1></div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="sidebar-title">Music Generator</div>', unsafe_allow_html=True)
    
    os.makedirs(abc_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer(tokenizer_path)

    st.sidebar.header("Input Music Sequence")
    start_it = st.sidebar.text_area("Enter the start of your music sequence (e.g., X:1...)", value="X:1\n")

    max_length = st.sidebar.slider("Max length of generated music", min_value=256, max_value=1024, value=512)
    # start_it = st.sidebar.text_input("", "")

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
                # st.markdown('<div class="generated-container">', unsafe_allow_html=True)
                # st.text_area("Generated Music", value=abc_string, height=400)
                # st.markdown('</div>', unsafe_allow_html=True)
                
                song_details = load_song_details(json_file_path)
                display_songs(song_details)
            else:
                st.error("The generated songs are not clean. Please try again.")
        else:
            st.markdown('<div class="upload-container"><p class="instructions">Please input a starting sequence to generate music.</p></div>', unsafe_allow_html=True)
    
    # Display existing songs
    # song_details = load_song_details(json_file_path)
    # display_songs(song_details)

if __name__ == "__main__":
    main()
