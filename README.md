# Digit Classification using Convolutional Neural Networks (CNN)

![Digit Classification](https://img.shields.io/badge/Digit%20Classification-CNN-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)
![MNIST Dataset](https://img.shields.io/badge/Dataset-MNIST-blue.svg)


A machine learning project for digit classification using Convolutional Neural Networks (CNN) and the MNIST dataset. This project showcases the power of CNNs in recognizing and classifying handwritten digits.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)


## Introduction
The Digit Classification project focuses on recognizing and classifying handwritten digits (0-9) using Convolutional Neural Networks (CNN). It employs a deep learning approach to process and classify the digits, making it a valuable tool for various applications, including optical character recognition (OCR) and digit identification.

## Features
- Input features consist of 28x28 pixel grayscale images of handwritten digits.
- Utilizes a deep learning CNN model for digit classification.
- Provides accuracy metrics and visualizations of the model's performance.

## Dataset

The dataset used in this project is the **MNIST dataset**, a widely recognized and benchmark dataset for digit classification tasks. Here are some important details about the dataset:

### Dataset Overview

- **Number of Samples:** The MNIST dataset consists of a total of 70,000 images.
  - **Training Set:** 60,000 images
  - **Test Set:** 10,000 images

- **Digit Classes:** It is a multi-class classification problem, where each image represents one of the ten digit classes (0-9).

- **Image Dimensions:** All images in the dataset are grayscale and have a fixed size of **28x28 pixels**.

- **Pixel Values:** The pixel values of the images are represented as grayscale values ranging from 0 (black) to 255 (white).


## Model Architecture

The digit classification model in this project utilizes a Convolutional Neural Network (CNN), a deep learning architecture particularly effective in image classification tasks. Here's a detailed overview of the CNN architecture:

### Convolutional Layers

- **Input Layer:** The input layer receives 28x28 pixel grayscale images of handwritten digits as input. Each image is treated as a 2D array of pixel values.

- **Convolutional Layers:** The core of the model consists of convolutional layers, each with its own set of learnable filters (kernels). These filters slide across the input image to extract various features such as edges, corners, and textures. Convolutional layers are equipped with ReLU (Rectified Linear Unit), to introduce non-linearity.

- **Pooling Layers:** After each convolutional layer, a Max-pooling layer is applied to reduce the spatial dimensions of the feature maps. 

### Fully Connected Layers

- **Flatten Layer:** Following the convolutional and pooling layers, the data is flattened into a 1D vector. This prepares the features for input into fully connected layers.

- **Output Layer:** The final layer of the model is the output layer. It contains ten neurons, each representing a digit class (0-9). The softmax activation function is applied to the output layer, producing probability scores for each class. The class with the highest probability is considered the predicted digit.


