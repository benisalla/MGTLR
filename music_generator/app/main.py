# import base64
# import streamlit as st
# from music_generator.app.utils import clear_directory, convert_midi_to_wav, convert_wav_to_mp3, generate_songs, generate_string, load_model, load_tokenizer
# import torch
# import os
# from glob import glob

# def add_background_image(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded_image = base64.b64encode(image_file.read()).decode()
#     with open("./music_generator/app/style/style.css", "r") as css_file:
#         css_content = css_file.read()
#         css_content = css_content.replace("{encoded_image}", encoded_image)
#     st.markdown(
#         f"<style>{css_content}</style>",
#         unsafe_allow_html=True
#     )
    
# def save_file(abc_content, file_path):
#     with open(file_path, 'w') as abc_file:
#         abc_file.write(abc_content)

# def display_mid_files(abc_dir):
#     mid_files = glob(os.path.join(abc_dir, "*.mid"))
#     if not mid_files:
#         st.warning("No MIDI files found in the directory.")
#     else:
#         cols = st.columns(2)  # Create two columns for the grid
#         for i, mid_file in enumerate(mid_files):
#             col = cols[i % 2]  # Alternate between columns
#             with col:
#                 file_name = os.path.basename(mid_file)
#                 wav_file = convert_midi_to_wav(mid_file)
#                 mp3_file = convert_wav_to_mp3(wav_file)
#                 audio_bytes = open(mp3_file, 'rb').read()
#                 st.audio(audio_bytes, format='audio/mp3')
#                 st.markdown(f"[Download {file_name.replace('.mid', '.mp3')}]({mp3_file})")
#                 abc_path = mid_file.replace('.mid', '.abc')
#                 try:
#                     with open(abc_path, 'r') as abc_file:
#                         abc_content = abc_file.read()
#                     with st.expander("Show ABC Annotations"):
#                         st.text(abc_content)
#                 except FileNotFoundError:
#                     st.error(f"ABC file not found for {file_name}")

# def main():
#     bk_img = "./music_generator/app/src/image.png"
#     tokenizer_path = "./music_generator/src/tokenizer/mgt_tokenizer_v1.model"  
#     checkpoint_path = './music_generator/src/checkpoints/mg_chpts_v1.pth' 
#     abc_dir = "./music_generator/app/abc_dir"
    
#     clear_directory(abc_dir)
#     add_background_image(bk_img)
    
#     st.markdown('<div class="title-container"><h1 class="title">🎵 Music Generator App</h1></div>', unsafe_allow_html=True)
#     st.markdown('<div class="upload-container"><p class="instructions">Input a starting sequence, and the model will generate music for you.</p></div>', unsafe_allow_html=True)
    
#     st.sidebar.markdown('<div class="sidebar-title">Music Generator</div>', unsafe_allow_html=True)
    
#     abc_dir = "generated_songs"
#     os.makedirs(abc_dir, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = load_model(checkpoint_path, device)
#     tokenizer = load_tokenizer(tokenizer_path)

#     st.sidebar.header("Input Music Sequence")
#     start_sequence = st.sidebar.text_area("Enter the starting sequence of notes (e.g., <SOS>X:1...)", value="<SOS>X:1\n")

#     max_length = st.sidebar.slider("Max length of generated music", min_value=256, max_value=1024, value=512)
#     start_it = st.sidebar.text_input("Enter the start of your music sequence", "")

#     if st.sidebar.button("Generate Music"):
#         if start_sequence:
#             in_str = f"<SOS>{start_it}"
#             abc_string = generate_string(in_str, model, tokenizer, device, max_new_tokens=max_length, temperature=1.0)
            
#             if not isinstance(abc_string, str):
#                 st.error("Generated string is not valid. Please try again.")
#                 return
            
#             is_songs_exist = generate_songs(abc_string, abc_dir)
            
#             if is_songs_exist:
#                 st.markdown('<div class="generated-container">', unsafe_allow_html=True)
#                 st.text_area("Generated Music", value=abc_string, height=400)
#                 st.markdown('</div>', unsafe_allow_html=True)
#                 display_mid_files(abc_dir)
#             else:
#                 st.error("The generated songs are not clean. Please try again.")
#         else:
#             st.markdown('<div class="upload-container"><p class="instructions">Please input a starting sequence to generate music.</p></div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

# # poetry run streamlit run music_generator/app/main.py




















# import base64
# import streamlit as st
# from music_generator.app.utils import clear_directory, convert_midi_to_wav, convert_wav_to_mp3, generate_songs, generate_string, load_model, load_tokenizer, save_file
# import torch
# import os
# import json
# from glob import glob
# from datetime import datetime

# def add_background_image(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded_image = base64.b64encode(image_file.read()).decode()
#     with open("./music_generator/app/style/style.css", "r") as css_file:
#         css_content = css_file.read()
#         css_content = css_content.replace("{encoded_image}", encoded_image)
#     st.markdown(
#         f"<style>{css_content}</style>",
#         unsafe_allow_html=True
#     )
    

# # def display_mid_files(abc_dir, song_details):
# #     mid_files = glob(os.path.join(abc_dir, "*.mid"))
# #     if not mid_files:
# #         st.warning("No MIDI files found in the directory.")
# #     else:
# #         cols = st.columns(2)  # Create two columns for the grid
# #         for i, mid_file in enumerate(mid_files):
# #             col = cols[i % 2]  # Alternate between columns
# #             with col:
# #                 file_name = os.path.basename(mid_file)
# #                 wav_file = convert_midi_to_wav(mid_file)
# #                 mp3_file = convert_wav_to_mp3(wav_file)
# #                 audio_bytes = open(mp3_file, 'rb').read()
# #                 st.audio(audio_bytes, format='audio/mp3')
# #                 st.markdown(f"[Download {file_name.replace('.mid', '.mp3')}]({mp3_file})")
# #                 abc_path = mid_file.replace('.mid', '.abc')
# #                 try:
# #                     with open(abc_path, 'r') as abc_file:
# #                         abc_content = abc_file.read()
# #                     with st.expander("Show ABC Annotations"):
# #                         st.text(abc_content)
# #                     # Save song details to dictionary
# #                     song_details[file_name] = {
# #                         "abc": abc_content,
# #                         "mp3_path": mp3_file
# #                     }
# #                 except FileNotFoundError:
# #                     st.error(f"ABC file not found for {file_name}")



# def display_mid_files(abc_dir, song_details):
#     mid_files = glob(os.path.join(abc_dir, "*.mid"))
#     if not mid_files:
#         st.warning("No MIDI files found in the directory.")
#     else:
#         cols = st.columns(2)  # Create two columns for the grid
#         for i, mid_file in enumerate(mid_files):
#             col = cols[i % 2]  # Alternate between columns
#             with col:
#                 file_name = os.path.basename(mid_file)
#                 wav_file = convert_midi_to_wav(mid_file)
#                 mp3_file = convert_wav_to_mp3(wav_file)
#                 audio_bytes = open(mp3_file, 'rb').read()
#                 st.audio(audio_bytes, format='audio/mp3')
#                 st.markdown(f"[Download {file_name.replace('.mid', '.mp3')}]({mp3_file})")
#                 abc_path = mid_file.replace('.mid', '.abc')
#                 try:
#                     with open(abc_path, 'r') as abc_file:
#                         abc_content = abc_file.read()
#                     with st.expander("Show ABC Annotations"):
#                         st.text(abc_content)
#                     # Save song details to dictionary
#                     song_details[file_name] = {
#                         "abc": abc_content,
#                         "mp3_path": mp3_file
#                     }
#                 except FileNotFoundError:
#                     st.error(f"ABC file not found for {file_name}")
#                 # Delete the MIDI file after conversion
#                 os.remove(mid_file)




# def main():
#     bk_img = "./music_generator/app/src/image.png"
#     tokenizer_path = "./music_generator/src/tokenizer/mgt_tokenizer_v1.model"  
#     checkpoint_path = './music_generator/src/checkpoints/mg_chpts_v1.pth' 
#     abc_dir = "./music_generator/app/abc_dir"
    
#     clear_directory(abc_dir)
#     add_background_image(bk_img)
    
#     st.markdown('<div class="title-container"><h1 class="title">🎵 Music Generator App</h1></div>', unsafe_allow_html=True)
#     st.markdown('<div class="upload-container"><p class="instructions">Input a starting sequence, and the model will generate music for you.</p></div>', unsafe_allow_html=True)
    
#     st.sidebar.markdown('<div class="sidebar-title">Music Generator</div>', unsafe_allow_html=True)
    
#     abc_dir = "generated_songs"
#     os.makedirs(abc_dir, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = load_model(checkpoint_path, device)
#     tokenizer = load_tokenizer(tokenizer_path)

#     st.sidebar.header("Input Music Sequence")
#     start_sequence = st.sidebar.text_area("Enter the starting sequence of notes (e.g., <SOS>X:1...)", value="<SOS>X:1\n")

#     max_length = st.sidebar.slider("Max length of generated music", min_value=256, max_value=1024, value=512)
#     start_it = st.sidebar.text_input("Enter the start of your music sequence", "")

#     if st.sidebar.button("Generate Music"):
#         if start_sequence:
#             in_str = f"<SOS>{start_it}"
#             abc_string = generate_string(in_str, model, tokenizer, device, max_new_tokens=max_length, temperature=1.0)
            
#             if not isinstance(abc_string, str):
#                 st.error("Generated string is not valid. Please try again.")
#                 return
            
#             is_songs_exist = generate_songs(abc_string, abc_dir)
            
#             if is_songs_exist:
#                 st.markdown('<div class="generated-container">', unsafe_allow_html=True)
#                 st.text_area("Generated Music", value=abc_string, height=400)
#                 st.markdown('</div>', unsafe_allow_html=True)
#                 # Save the ABC content
#                 curr_dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#                 abc_file_path = os.path.join(abc_dir, f'abs_song_{curr_dtime}.abc')
#                 save_file(abc_string, abc_file_path)
                
#                 # Dictionary to hold song details
#                 song_details = {}
#                 display_mid_files(abc_dir, song_details)
                
#                 # Save song details to JSON file
#                 json_file_path = os.path.join(abc_dir, 'song_details.json')
#                 with open(json_file_path, 'w') as json_file:
#                     json.dump(song_details, json_file, indent=4)
#             else:
#                 st.error("The generated songs are not clean. Please try again.")
#         else:
#             st.markdown('<div class="upload-container"><p class="instructions">Please input a starting sequence to generate music.</p></div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

# # poetry run streamlit run music_generator/app/main.py

















# import base64
# import streamlit as st
# from music_generator.app.utils import clear_directory, convert_midi_to_wav, convert_wav_to_mp3, generate_string, load_model, load_tokenizer
# import torch
# import os
# import json
# import re
# from glob import glob
# from music21 import converter
# from datetime import datetime
# from midi2audio import FluidSynth
# from pydub import AudioSegment

# def add_background_image(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded_image = base64.b64encode(image_file.read()).decode()
#     with open("./music_generator/app/style/style.css", "r") as css_file:
#         css_content = css_file.read()
#         css_content = css_content.replace("{encoded_image}", encoded_image)
#     st.markdown(
#         f"<style>{css_content}</style>",
#         unsafe_allow_html=True
#     )
    
# def save_abc_file(abc_content, file_path):
#     with open(file_path, 'w') as abc_file:
#         abc_file.write(abc_content)

# def convert_midi_to_wav(mid_file):
#     fs = FluidSynth()
#     wav_file = mid_file.replace('.mid', '.wav')
#     fs.midi_to_audio(mid_file, wav_file)
#     return wav_file

# def convert_wav_to_mp3(wav_file):
#     mp3_file = wav_file.replace('.wav', '.mp3')
#     audio = AudioSegment.from_wav(wav_file)
#     audio.export(mp3_file, format='mp3')
#     os.remove(wav_file)
#     return mp3_file

# def save_song_details(song_details, json_file_path):
#     with open(json_file_path, 'w') as json_file:
#         json.dump(song_details, json_file, indent=4)

# def load_song_details(json_file_path):
#     if os.path.exists(json_file_path):
#         with open(json_file_path, 'r') as json_file:
#             return json.load(json_file)
#     return {}

# def display_songs(song_details):
#     cols = st.columns(2)  # Create two columns for the grid
#     for i, (file_name, details) in enumerate(song_details.items()):
#         col = cols[i % 2]  # Alternate between columns
#         with col:
#             st.markdown(f"### {file_name}")
#             audio_bytes = open(details['mp3_path'], 'rb').read()
#             st.audio(audio_bytes, format='audio/mp3')
#             st.markdown(f"[Download {file_name}]({details['mp3_path']})")
#             with st.expander("Show ABC Annotations"):
#                 st.text(details['abc'])

# def generate_songs(abc_string, abc_dir, json_file_path):
#     pattern = re.compile(r'<SOS>(.*?)<EOS>', re.DOTALL)
#     matches = pattern.findall(abc_string)
#     abc_strings = [match.strip() for match in matches if match.strip()]

#     os.makedirs(abc_dir, exist_ok=True)
    
#     is_exist = False
#     song_details = load_song_details(json_file_path)
    
#     for i, match in enumerate(abc_strings):
#         curr_dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         try:
#             song = converter.parse(match, format='abc')
#             midi_file_name = f'abs_song_{curr_dtime}_{i + 1}.mid'
#             midi_path = os.path.join(abc_dir, midi_file_name)
#             song.write('midi', fp=midi_path)
#             is_exist = True

#             # Convert MIDI to WAV and then to MP3
#             wav_file = convert_midi_to_wav(midi_path)
#             mp3_file = convert_wav_to_mp3(wav_file)
#             os.remove(midi_path)  # Delete MIDI file after conversion

#             # Save song details
#             song_details[midi_file_name] = {
#                 "abc": match,
#                 "mp3_path": mp3_file
#             }

#         except Exception as e:
#             pass

#     save_song_details(song_details, json_file_path)
#     return is_exist

# def main():
#     bk_img = "./music_generator/app/src/image.png"
#     tokenizer_path = "./music_generator/src/tokenizer/mgt_tokenizer_v1.model"  
#     checkpoint_path = './music_generator/src/checkpoints/mg_chpts_v1.pth' 
#     abc_dir = "./music_generator/app/abc_dir"
#     json_file_path = os.path.join(abc_dir, 'song_details.json')
    
#     clear_directory(abc_dir)
#     add_background_image(bk_img)
    
#     st.markdown('<div class="title-container"><h1 class="title">🎵 Music Generator App</h1></div>', unsafe_allow_html=True)
#     st.markdown('<div class="upload-container"><p class="instructions">Input a starting sequence, and the model will generate music for you.</p></div>', unsafe_allow_html=True)
    
#     st.sidebar.markdown('<div class="sidebar-title">Music Generator</div>', unsafe_allow_html=True)
    
#     abc_dir = "generated_songs"
#     os.makedirs(abc_dir, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = load_model(checkpoint_path, device)
#     tokenizer = load_tokenizer(tokenizer_path)

#     st.sidebar.header("Input Music Sequence")
#     start_sequence = st.sidebar.text_area("Enter the starting sequence of notes (e.g., <SOS>X:1...)", value="<SOS>X:1\n")

#     max_length = st.sidebar.slider("Max length of generated music", min_value=256, max_value=1024, value=512)
#     start_it = st.sidebar.text_input("Enter the start of your music sequence", "")

#     if st.sidebar.button("Generate Music"):
#         if start_sequence:
#             in_str = f"<SOS>{start_it}"
#             abc_string = generate_string(in_str, model, tokenizer, device, max_new_tokens=max_length, temperature=1.0)
            
#             if not isinstance(abc_string, str):
#                 st.error("Generated string is not valid. Please try again.")
#                 return
            
#             curr_dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             abc_file_path = os.path.join(abc_dir, f'abs_song_{curr_dtime}.abc')
#             save_abc_file(abc_string, abc_file_path)
            
#             is_songs_exist = generate_songs(abc_string, abc_dir, json_file_path)
            
#             if is_songs_exist:
#                 st.markdown('<div class="generated-container">', unsafe_allow_html=True)
#                 st.text_area("Generated Music", value=abc_string, height=400)
#                 st.markdown('</div>', unsafe_allow_html=True)
                
#                 song_details = load_song_details(json_file_path)
#                 display_songs(song_details)
#             else:
#                 st.error("The generated songs are not clean. Please try again.")
#         else:
#             st.markdown('<div class="upload-container"><p class="instructions">Please input a starting sequence to generate music.</p></div>', unsafe_allow_html=True)
    
#     # Display existing songs
#     song_details = load_song_details(json_file_path)
#     display_songs(song_details)

# if __name__ == "__main__":
#     main()

# # poetry run streamlit run music_generator/app/main.py
















import base64
import streamlit as st
from music_generator.app.utils import clear_directory, convert_midi_to_wav, convert_wav_to_mp3, generate_songs, generate_string, load_model, load_song_details, load_tokenizer, save_abc_file
import torch
import os
import json
import re
from glob import glob
from music21 import converter
from datetime import datetime
from midi2audio import FluidSynth
from pydub import AudioSegment

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
    cols = st.columns(2)  # Create two columns for the grid
    for i, (file_name, details) in enumerate(song_details.items()):
        col = cols[i % 2]  # Alternate between columns
        with col:
            st.markdown(f"### {file_name}")
            audio_bytes = open(details['mp3_path'], 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')
            st.markdown(f"[Download {file_name}]({details['mp3_path']})")
            with st.expander("Show ABC Annotations"):
                st.text(details['abc'])

def main():
    bk_img = "./music_generator/app/src/image.png"
    tokenizer_path = "./music_generator/src/tokenizer/mgt_tokenizer_v1.model"  
    checkpoint_path = './music_generator/src/checkpoints/mg_chpts_v1.pth' 
    abc_dir = "./music_generator/app/abc_dir"
    json_file_path = os.path.join(abc_dir, 'song_details.json')
    
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
    song_details = load_song_details(json_file_path)
    display_songs(song_details)

if __name__ == "__main__":
    main()
