# Handwritten-Signature-Verification
Create a machine learning model capable of distinguishing between genuine and forged handwritten signatures.

# Signature Verification System using Siamese Neural Networks

## Project Overview

This project implements a signature verification system using a Siamese Neural Network architecture with a ResNet backbone and a spatial attention mechanism. The system is designed to verify whether two signature images are genuine or forged by comparing their feature representations. The project includes data preparation, model implementation, training, and evaluation components.

## Table of Contents


- [Project Overview](#project-overview)
- [Dataset Preparation](#dataset-preparation)

- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)

- [Loss Functions](#loss-functions)
- [Training](#training)

- [Evaluation](#evaluation)
- [Usage](#usage)

- [Dependencies](#dependencies)
- [Installation](#installation)

- [Conclusion](#conclusion)

## Dataset Preparation

### Random Image Selection

The `random_images` function is used to collect paths of random images from the dataset folder. This folder contains subfolders, each representing a category or signature in the dataset.

### Triplet Dataset Preparation

The `triplet_dataset_preparation` function prepares a dataset of triplets (anchor, positive, negative) for training the model using the triplet loss. It identifies genuine and forged images based on the folder structure and filenames.

### Duplet Dataset Preparation

The `duplet_dataset_preparation` function prepares a dataset of duplets (image pairs) for training a logistic regression model for signature verification. It creates pairs of genuine and mixed (genuine and forged) images.

## Preprocessing

### Image Preprocessing

The `preprocess_image` function preprocesses an image for signature verification tasks. It includes steps like conversion to grayscale, noise reduction, morphological operations, thresholding, bounding box extraction, and padding.

### Test Image Preprocessing

The `test_preprocess_image` function preprocesses images for testing, enhancing robustness against minor rotations and distortions.

## Model Architecture

### Spatial Attention

The `SpatialAttention` class implements a spatial attention mechanism for convolutional neural networks. It dynamically weights feature channels in a feature map based on their importance.

### Siamese Network with ResNet Backbone

The `SiameseResNet` class implements a Siamese network with a ResNet backbone and spatial attention modules. The network processes two input images and extracts feature vectors for comparison.

## Loss Functions

### Triplet Loss

The `TripletLoss` class implements the triplet loss function, which encourages the network to embed similar images closer together in the feature space while pushing dissimilar images further apart.

## Training

### Training the Siamese Model

The `train_siamese_model` function trains a Siamese network model for signature verification using the triplet loss.

### Logistic Regression on Siamese Features

The `LogisticSiameseRegression` class implements a logistic regression classifier on top of a pre-trained Siamese network. The `train_model` function trains this model using duplet data.

## Evaluation

### Accuracy Calculation

The `calculate_accuracy` function evaluates the model's performance on a dataset by comparing predictions with ground truth labels.

### Confusion Matrix

The `build_confusion_matrix` function creates and plots a normalized confusion matrix for model evaluation.

### Predictions for Single Pairs

The `get_predictions_for_single` function predicts the similarity of a single pair of signature images using a trained model.

### Voting Mechanism

The `perform_voting` function performs majority voting on model predictions for a dataset of genuine and test image pairs.

## Usage


1. **Data Preparation:**
   - Organize your dataset into folders representing different signature classes, with subfolders for genuine and forged images.
   - Use the provided functions to prepare triplet and duplet datasets.


2. **Preprocessing:**
   - Preprocess your images using the `preprocess_image` and `test_preprocess_image` functions.


3. **Model Training:**
   - Train the Siamese network using the `train_siamese_model` function.
   - Train the logistic regression model using the `train_model` function.


4. **Evaluation:**
   - Evaluate the model's performance using the `calculate_accuracy`, `build_confusion_matrix`, and `get_predictions_for_single` functions.
   - Use the voting mechanism for improved predictions.

## Dependencies

- Python 3.x
- PyTorch
- pandas
- PIL (Pillow)
- matplotlib
- scikit-learn
- numpy

## Installation

To install the required dependencies, run:

```bash
pip install torch pandas pillow matplotlib scikit-learn numpy
```

## Conclusion
This project provides a comprehensive framework for building a signature verification system using a Siamese network with a ResNet backbone and spatial attention mechanism. It includes all necessary steps from data preparation and preprocessing to model training and evaluation. The use of an NVIDIA L4 GPU ensures efficient training and inference, making the system suitable for real-world applications.
