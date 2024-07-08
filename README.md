<div align="center">
  <img src="https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/035598be-ea1c-4501-947a-ff51524e78ef" width="200" height="200"/>
  <h1>MG-Transformer: Music Generation Transformer from Scratch</h1>
  <p>Implementing a Music Generation Transformer model from scratch.</p>
</div>

---

## Table of Contents 📘
- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Fine-Tuning](#fine-tuning)
- [Model Performance](#model-performance)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [About Me](#about-me)

---

## About The Project

<div align="center">
  <img src="https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/4935a979-a8e8-40f0-8ffb-e621025aac2f" width="600" height="300"/>
</div>

MG-Transformer offers a minimalist, yet complete implementation of the Music Generation Transformer (MG-Transformer) architecture for music generation tasks. This project aims to provide a clear and structured approach to building Transformers for music generation, making it accessible for educational purposes and practical applications alike.

---

## Features

- **Modular Design**: Clear separation of components like data processing, model architecture, and training routines.
- **Customizable**: Easy to adapt the architecture and data pipeline for various datasets and applications.
- **Poetry Dependency Management**: Utilizes Poetry for simple and reliable package management.

---

## Project Structure
```
MG-TRANSFORMER
│
├── generated_songs           # Generated music files
├── music_generator           # Main project directory
│   ├── app                   # Application files
│   ├── core                  # Core configurations and caching
│   ├── data                  # Data processing modules
│   ├── model                 # Transformer model components
│   └── src                   # Source files
│       ├── checkpoints       # Model checkpoints
│       ├── dataset           # Dataset handling
│       └── tokenizer         # Tokenizer modules
├── tests                     # Test scripts
│   ├── model                 # Model tests
│   └── tokenizer             # Tokenizer tests
├── tokenizing                # Tokenizing scripts
│   └── tokenizer             # Tokenizer implementation
│       ├── __init__.py       # Initializer for tokenizer
│       ├── KerasTokenizer.py # Keras Tokenizer implementation
│       ├── MGTokenizer.py    # MG Tokenizer implementation
│       ├── NaiveTokenizer.py # Naive Tokenizer implementation
│       ├── train_keras_tokenizer.py # Script to train Keras tokenizer
│       └── train_mg_tokenizer.py    # Script to train MG tokenizer
├── finetune.py               # Script for fine-tuning the model
├── train.py                  # Script to train the model
├── .gitignore                # Git ignore file
└── README.md                 # Project README file

```

---

### Built With
This section should list any major frameworks/libraries used to bootstrap your project:
- [PyTorch](https://pytorch.org/)

---

## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/benisalla/mg-transformer.git
   ```
2. Install Poetry packages
   ```sh
   poetry install
   ```

---

## Usage

How you can use this code

### Training

To train the model using the default configuration:

```bash
poetry run python train.py
```

### Fine-Tuning

To fine-tune a pre-trained model:

```bash
poetry run python finetune.py
```

---

## Model Performance

The MG-Transformer model was evaluated on a comprehensive set of test music data to gauge its accuracy and performance. Here are the results:

- **Accuracy on test music data**: 81.60%

These results demonstrate the effectiveness of the MG-Transformer model in handling complex music generation tasks. We continuously seek to improve the model and update the metrics as new test results become available.

![image](https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/62531c3f-6684-4000-a151-acee6a399ab3)

![image](https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/7aafbd0a-f48b-46dd-9caf-d99f42e063e3)

---

## Roadmap

See the [open issues](https://github.com/benisalla/mg-transformer/issues) for a list of proposed features (and known issues).

---

## Contributing 

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is made available under **fair use guidelines**. While there is no formal license associated with the repository, users are encouraged to credit the source if they utilize or adapt the code in their work. This approach promotes ethical practices and contributions to the open-source community. For citation purposes, please use the following:

```bibtex
@misc{mg_transformer_2024,
  title={MG-Transformer: Music Generation Transformer from Scratch},
  author={Ben Alla Ismail},
  year={2024},
  url={https://github.com/benisalla/mg-transformer}
}
```

---

## About Me

🎓 **Ismail Ben Alla** - Neural Network Enthusiast

I am deeply passionate about exploring artificial intelligence and its potential to solve complex problems and unravel the mysteries of our universe. My academic and professional journey is characterized by a commitment to learning and innovation in AI, deep learning, and machine learning.

### What Drives Me
- **Passion for AI**: Eager to push the boundaries of technology and discover new possibilities.
- **Continuous Learning**: Committed to staying informed and skilled in the latest advancements.
- **Optimism and Dedication**: Motivated by the challenges and opportunities that the future of AI holds.

I thoroughly enjoy what I do and am excited about the future of AI and machine learning. Let's connect and explore the endless possibilities of artificial intelligence together!

<div align="center">
  <a href="https://twitter.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
  <a href="https://linkedin.com/in/ismail-ben-alla-7144b5221/" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="ismail-ben-alla-7144b5221/" height="30" width="40" />
  </a>
  <a href="https://instagram.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
</div>

---

<div align="center">
  <h4>Get ready to see music transform into a symphony 🎵✨🎶</h4>
  <img src="https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/087e0049-d113-4df6-8fb3-183ebc4f85e1" width="500" height="300"/>
</div>