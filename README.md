# music_generator_with_3_nlp_algorithms
I have built in This project a music generator in this repository using RNNs, LSTMs, and transformers in order to learn more about their structure and performance.


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The Music Generator with ABC Annotations is a project designed to create music using the ABC notation format. It leverages the power of Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), and transformers to generate musical compositions. 

We have used these neural network architectures to learn more about them and compare their performance and usage in the context of music generation.




## Features

- Music generation in the ABC notation format.
- Utilization of RNN, LSTM, and transformers for music composition.
- Application of the same approach used for processing ABC annotations to solve various NLP problems.




## Reuse

To reuse these notebooks, you can follow these steps or tips:

1. Clone this repository or fork it to your own repository.
2. Replace the dataset with your own, whether it's voices, text, or any sequential problem.
3. Perform hyperparameter tuning to improve performance further.
4. If you make use of these notebooks, don't forget to give us a mention! 😁😂

## Dataset

This dataset serves as the foundation for the Music Generator with ABC Annotations project, containing a rich collection of musical compositions in ABC notation. Harnessing the might of RNN, LSTM, and transformers, it fuels the creation of entirely new musical masterpieces through advanced model training and generation techniques.

   our data set contains : 
   - train dataset
   - test dataset
   - validation dataset
   each one contains songs or samples with this format :
   ```
   X:1
   L:1/8
   M:4/4
   K:Emin
   |: E2 EF E2 EF | DEFG AFDF | E2 EF E2 B2 |1 efe^d e2 e2 :|2 efe^d e3 B |: e2 ef g2 fe |
   defg afdf |1 e2 ef g2 fe | efe^d e3 B :|2 g2 bg f2 af | efe^d e2 e2 ||
   ```




## Training Results

Below are the training results for the different models we utilized:

   - **Recurrent Neural Networks (RNNs):**

   <p align="center">
      <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272082340-26f21583-467f-4875-a862-cc9beff48571.png" height="300" width="500"/>
   </p>

   - **Long Short-Term Memory Networks (LSTMs):**

   <p align="center">
      <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272082328-72f33452-4e3b-41fc-836c-837d62e9fcc7.png" height="300" width="500"/>
   </p>

   - **Beloved Model 😍 (Transformers):**

   <p align="center">
      <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272082342-4e32381d-c90f-4a71-b740-6c20cd072ed9.png" height="300" width="500"/>
   </p>







## Inference Examples

Below are video examples showcasing the inference results of the different models we utilized:

### Recurrent Neural Networks (RNNs)

<p align="center">
   <video controls width="500" height="300">
       <source src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272085172-0c75e0a5-0f15-4657-9c70-24d0f501eec6.mp4" type="video/mp4">
       Your browser does not support the video tag.
   </video>
</p>

### Long Short-Term Memory Networks (LSTMs)

<p align="center">
   [Add Video Link for LSTM Inference]
</p>

### Beloved Model 😍 (Transformers)

<p align="center">
   [Add Video Link for Transformer Inference]
</p>






## Contributing

We welcome contributions to this project. If you would like to contribute, please follow our [Contribution Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [Your License Name] - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to acknowledge the following for their contributions and inspiration:
