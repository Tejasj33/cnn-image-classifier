# CNN Image Classifier (CIFAR-10 Dataset)

This project involves building a Convolutional Neural Network (CNN) model to classify images from the CIFAR-10 dataset. The model is built using TensorFlow and Keras, trained on the CIFAR-10 dataset, and can predict the class of a given image.

## Problem Statement

The goal of this project is to create a model that can accurately classify images into one of 10 categories from the CIFAR-10 dataset.

## Dataset Used: CIFAR-10

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Model Architecture

1. **Convolutional Layer 1**: 32 filters of size (3x3), activation function ReLU
2. **Max-Pooling Layer 1**: Pool size (2x2)
3. **Convolutional Layer 2**: 64 filters of size (3x3), activation function ReLU
4. **Max-Pooling Layer 2**: Pool size (2x2)
5. **Convolutional Layer 3**: 128 filters of size (3x3), activation function ReLU
6. **Max-Pooling Layer 3**: Pool size (2x2)
7. **Flatten Layer**: Converts 3D output into 1D vector
8. **Fully Connected Layer (Dense)**: 128 neurons, ReLU activation
9. **Output Layer**: 10 neurons (one for each class), softmax activation

## Training Results

- **Test Accuracy**: 71.48%
- **Training Accuracy Plot**: [Insert Plot Here]
- **Validation Accuracy Plot**: [Insert Plot Here]

## How to Run the Code

1. **Clone the repository**:
