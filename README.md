# Speech Emotion Classification Deep Learning Project

## Overview

This project focuses on the classification of speech emotions using deep learning techniques. The goal is to train a Convolutional Neural Network (CNN) on the Toronto Emotional Speech Set (TESS) dataset to accurately predict the emotional content of speech samples.

## Dataset

The dataset used in this project is the Toronto Emotional Speech Set (TESS), which can be found at the following link: [TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

## Preprocessing

The audio data undergoes the following preprocessing steps:

1. **Loading Data**: The audio files are loaded using the Librosa library.
2. **Feature Extraction**: Mel-frequency cepstral coefficients (MFCCs) are extracted from the audio signals. This process captures relevant features for emotion classification.
3. **Label Extraction**: Emotion labels are extracted from the filenames of the audio files.

## CNN Architecture

The Convolutional Neural Network architecture used for this project consists of the following layers:

1. **Convolutional Layers**: Two sets of convolutional layers with rectified linear unit (ReLU) activation functions.
2. **Max Pooling Layers**: Max pooling layers are applied to downsample the spatial dimensions of the convolutional layers.
3. **Flatten Layer**: Flattens the output of the convolutional layers to be fed into the dense layers.
4. **Dense Layers**: Two dense layers with ReLU activation functions.
5. **Output Layer**: Dense layer with softmax activation for multi-class classification.

This architecture is designed to effectively learn and extract features from the MFCCs for accurate emotion classification.

## How to Use

1. **Dataset**: Download the TESS dataset from the provided link.
2. **Preprocessing**: Run the `preprocess_data.ipynb` notebook to preprocess the audio data and generate a CSV file with features and labels.
3. **Training**: Run the `train_model.ipynb` notebook to train the CNN model on the preprocessed data.
4. **Prediction**: Use the trained model to predict emotions for new audio samples.

## Dependencies

- Python 3.x
- TensorFlow
- Librosa
- NumPy
- Pandas
- Matplotlib

Install dependencies using:
```bash
pip install tensorflow librosa numpy pandas matplotlib
