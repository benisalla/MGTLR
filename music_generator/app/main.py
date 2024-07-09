import base64
import time
import streamlit as st
from music_generator.app.utils import add_background_image, clear_directory, display_songs, generate_songs, generate_string, init_session_state, load_model, load_song_details, load_tokenizer, save_abc_file
import torch
import os
from datetime import datetime
from music_generator.app.HTMLS import SMILE_SPINNER


def main():
    init_session_state(st)
    
    bk_img = "./music_generator/app/src/music-bkgd.jpeg" 
    add_background_image(bk_img, st)
    
    abc_dir = "./music_generator/app/abc_dir"
    
    os.makedirs(abc_dir, exist_ok=True)
    
    if st.session_state.first_run:
        clear_directory(abc_dir)
        st.session_state.first_run = False 
    
    st.markdown('''
    <div class="title-container">
        <h1 class="title">🎵🎵 Music Generator 🎵🎵</h1>
    </div>''', unsafe_allow_html=True)

    st.sidebar.markdown('''
    <div class="sidebar-title-container">
        <h2 class="sidebar-title">Music Generator</h2>
    </div>''', unsafe_allow_html=True)

    model_choice = st.sidebar.radio("Choose a model type:", ['TRF', 'LSTM', 'RNN'], horizontal=True)
    
    # todo: change the model here
    
    st.sidebar.markdown('<hr class="hr-style">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header">Start Note For Generator</div>', unsafe_allow_html=True)
    start_it = st.sidebar.text_area(" ",
        value="",
        height=200,
        help="Start your ABC notation down below ",
        placeholder="X: 45149\nT: BenIsAlla Tune\nC: Composer Name\nM: 3/4\nL: 1/8\nK: Amin\nA, B, C E | D E F A | B A G E | A3 z |"
    )

    st.sidebar.markdown('<hr class="hr-style">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header">Adjust Model Settings</div>', unsafe_allow_html=True)
    max_new_tokens = st.sidebar.slider("Max length of generated tokens:", min_value=256, max_value=1024, value=512)
    temperature = st.sidebar.slider("Temperature:", min_value=0.4, max_value=1.0, value=0.96, step=0.01)
    top_k = st.sidebar.slider("top_k:", min_value=0, max_value=10, value=0, step=1)


    st.sidebar.markdown('<hr class="hr-style">', unsafe_allow_html=True)
    if st.sidebar.button(f"Generate With {model_choice}", type="secondary", use_container_width=True):
        spinner_holder = st.empty()
        spinner_holder = st.markdown(SMILE_SPINNER, unsafe_allow_html=True)
        
        clear_directory(abc_dir)
        in_str = f"<SOS>{start_it}"        
        abc_string = generate_string(in_str, st.session_state.model, st.session_state.tokenizer, st.session_state.device, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                
        if not isinstance(abc_string, str):
            st.error("Generated string is not valid. Please try again.")
            return
        
        is_songs_exist = generate_songs(abc_string, abc_dir, st.session_state.json_file_path)
        
        spinner_holder.empty()
        
        if is_songs_exist:                
            song_details = load_song_details(st.session_state.json_file_path)
            display_songs(song_details, st)
        else:
            error_message = """
            <div class="error-component">
                <p><strong>Oops!</strong> It looks like there was a hiccup with the ABC notation provided by the our model.</p>
                <p class="help-text">No worries, though! Please <strong>double-check your starting notes</strong>, make any needed tweaks, and give it another shot. Try different configurations for the model and change the starting notes you've used.</p>
                <p class="encouragement">We’re here to help you create beautiful music, so don’t hesitate to adjust and try again. We believe in your creativity!</p>
            </div>
            """
            st.markdown(error_message, unsafe_allow_html=True)

    else:
        song_details = load_song_details(st.session_state.json_file_path)
        if song_details:
            display_songs(song_details, st)  
        else:
            st.markdown('''
            <div class="input-section">
                <p class="input-instructions">Set a start For ABC annotation and Click Generate</p>
            </div>
            ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()