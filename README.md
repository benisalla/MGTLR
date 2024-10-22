<div align="center">
  <img src="https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/1cb68f56-61c5-4540-b934-3562d2f15a42" width="200" height="200"/>
  <h1>MGTLR: Music Generator using Transformer, LSTM, and RNN</h1>
  <p>Implementing a music generation model using Transformer, LSTM, and RNN architectures from scratch.</p>
</div>

---

## Table of Contents 📘
- [About The Project](#about-the-project)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Fine-Tuning](#fine-tuning)
- [Some Examples](#examples-of-songs-generated)
- [License](#license)
- [About Me](#about-me)

---

## About The Project

<div align="center">
  <h3>Interface of our app</h3>
  <img src="https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/a3eed552-ff00-4cd0-920a-9bd02175ba9e" width="600" height="300"/>
</div>

MGTLR offers a streamlined, yet comprehensive, implementation of music generation using Transformer, LSTM, and RNN architectures. This project is designed to provide a clear, structured approach to neural network development for music generation, making it suitable for both educational and practical applications.

The methodologies developed could be applied to other types of audio and songs by adapting the input of the transformer (specifically, the tokenizer) to new formats.

---

## Features

- **Modular Design**: Clear separation of components such as data processing, model architecture, and training scripts.
- **Visualization of Annotations**: Enables checking and testing annotations.
- **Download Capability**: Allows users to download their favorite generated songs.
- **Customizable**: Easily adapt the architecture and data pipelines for different datasets and applications.
- **Poetry for Dependency Management**: Utilizes Poetry for straightforward and dependable package management.

---




## Project Structure
```
MGTLR
│
├── generated_songs          
├── music_generator           
│   ├── app              
│   ├── core                
│   ├── data                
│   ├── model                 
│   │   ├── LSTM
│   │   ├── TRF
│   │   └── RNN
│   └── src                
│       ├── checkpoints     
│       ├── dataset           
│       └── tokenizer         
├── tests                    
│   ├── model                 
│   └── tokenizer             
├── tokenizing               
│   └── tokenizer           
│       ├── __init__.py       
│       ├── KerasTokenizer.py 
│       ├── MGTokenizer.py    
│       ├── NaiveTokenizer.py 
│       ├── train_keras_tokenizer.py 
│       └── train_mg_tokenizer.py    
├── finetune.py              
├── train.py                  
├── .gitignore                
└── README.md                 
```

---


## Getting Started

Follow these simple steps to get a local copy up and running.

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/benisalla/mg-transformer.git
   ```
2. Install dependencies using Poetry
   ```sh
   poetry install
   ```
3. Activate the Poetry shell to set up your virtual environment
   ```sh
   poetry shell
   ```

### Running the Application

  To launch the Streamlit application, execute the following command:

  ```sh
  poetry run streamlit run music_generator/app/main.py
  ```

### Training

#### How to Run Training

To train the model using the default configuration, execute the following command:

```sh
poetry run python train.py
```

#### Results of Training Different Models

1. **Transformer (TRF)**

   ![Transformer Training Results](https://github.com/user-attachments/assets/bcf83ee8-a9ef-4cdb-83d2-435f07a00f49)

2. **Recurrent Neural Network (RNN)**

   ![RNN Training Results](https://github.com/user-attachments/assets/49a4e22b-5851-4817-91de-311b398c330d)

3. **Long Short-Term Memory (LSTM)**

   ![LSTM Training Results](https://github.com/user-attachments/assets/72b4d3fb-3cf4-444e-bbc4-02695c3c117c)



### Fine-Tuning

  To fine-tune a pre-trained model:

  ```sh
  poetry run python finetune.py
  ```

---



### Examples of Songs Generated

Here are some examples of songs generated by the MGTLR model:

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/cdc9c03f-4a22-4776-98ae-71d532800ff6

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/93f4042f-d17a-4382-8d0a-7e54632d321c

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/ae2c34ea-89cc-4bc8-aaf5-88d92422a8d1

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/f1cb56e8-1963-4234-9087-fff7cb875333

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/3c882167-ec4a-4f57-811f-1725255e4da7

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/f1eeb63e-0b8e-45db-b26e-0ddb39ad65d6

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/f2175c77-c903-415c-be10-69fd5248ed65

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/0001222a-20cf-4f1b-85b5-d06d77c7398e

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/338f7ff4-f0e6-47fd-a5b4-55ec13e28736

https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/eeccf4e8-d8a2-45f9-9cfe-bec4bf6bfbcf



---

## License

This project is made available under **fair use guidelines**. While there is no formal license associated with the repository, users are encouraged to credit the source if they utilize or adapt the code in their work. This approach promotes ethical practices and contributions to the open-source community. For citation purposes, please use the following:

```bibtex
@misc{mg_transformer_2024,
  title={MGTLR: Music Generator using Transformer, LSTM, and RNN},
  author={Ben Alla Ismail},
  year={2024},
  url={https://github.com/benisalla/mg-transformer}
}
```

---



## About Me

🎓 **Ismail Ben Alla** - Neural Network Enthusiast

As a dedicated advocate for artificial intelligence, I am deeply committed to exploring its potential to address complex challenges and to further our understanding of the universe. My academic and professional pursuits reflect a relentless dedication to advancing knowledge in AI, deep learning, and machine learning technologies.

### Core Motivations
- **Innovation in AI**: Driven to expand the frontiers of technology and unlock novel insights.
- **Lifelong Learning**: Actively engaged in mastering the latest technological developments.
- **Future-Oriented Vision**: Fueled by the transformative potential and future prospects of AI.

I am profoundly passionate about my work and optimistic about the future contributions of AI and machine learning. 

**Let's connect and explore the vast potential of artificial intelligence together!**
<div align="center">
  <a href="https://twitter.com/ismail_ben_alla" target="_blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="ismail_ben_alla" height="60" width="60" />
  </a>
  
  <a href="https://linkedin.com/in/ismail-ben-alla-7144b5221/" target="_blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="LinkedIn Profile" height="60" width="60"/>
  </a>
  
  <a href="https://instagram.com/ismail_ben_alla" target="_blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="Instagram Profile" height="60" width="60" />
  </a>
</div>

---

<div align="center">
  <h4>🎵✨🎶 Hit play and let the magic begin—watch notes turn into symphonies! 🎵✨🎶</h4>
  <img src="https://github.com/benisalla/music_generator_with_3_nlp_algorithms/assets/89405673/bd46df2e-e267-41af-8e6f-558cc7eac38b" width="500" height="400"/>
</div>
