# Digit Recognizer using Deep Learning

This project focuses on building a digit recognizer using a neural network implemented with TensorFlow and Keras. The dataset used is the well-known MNIST dataset of handwritten digits.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Results](#results)
7. [Custom Components](#custom-components)
8. [Visualization](#visualization)
9. [Conclusion](#conclusion)

## Project Overview
The goal of this project is to classify images of handwritten digits (0-9) using a deep neural network. The network is built without using Convolutional Neural Networks (CNNs), but by employing custom activation functions, initializers, regularizers, and constraints.

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Scipy
- Scikit-learn

You can install these libraries using pip:
```bash
pip install tensorflow pandas numpy matplotlib scipy scikit-learn
```

## Data Preparation
1. **Load the data**: The dataset is loaded from a CSV file containing 42,000 samples of handwritten digits.
2. **Normalization**: The pixel values of the images are normalized by dividing by 255.
3. **Reshape the data**: The training data is reshaped to have the shape (-1, 28, 28, 1).
4. **One-Hot Encoding**: The labels are one-hot encoded into 10 categories.

## Model Architecture
The model is built using a Sequential API in Keras with the following layers:
1. Input Layer
2. Flatten Layer
3. Dense Layer with SELU activation
4. Lambda Layer for standardization
5. PReLU Layer
6. Custom Dense Layer with softplus activation
7. Output Layer with softmax activation

## Training the Model
The model is compiled with the Adam optimizer and categorical cross-entropy loss. It is trained over 10 epochs with a batch size of 64.

## Results
After training, the model achieves high accuracy on the validation set. The performance can be visualized using accuracy and loss plots over the training epochs.

## Custom Components
The project includes custom implementations of:
1. **Activation Functions**: Custom softplus and SELU activation functions.
2. **Initializers**: Custom initializers for dense layers.
3. **Regularizers**: Custom L2 regularization.
4. **Constraints**: Custom constraints on the weights.

## Visualization
Visualizations include:
1. **Training History**: Plots of training and validation accuracy and loss.
2. **Sample Predictions**: Visualization of sample predictions with true labels.

## Conclusion
This project demonstrates the capability of building an effective digit recognizer using custom neural network components without relying on CNNs. Future work can include exploring different architectures and optimization techniques to further improve accuracy.
