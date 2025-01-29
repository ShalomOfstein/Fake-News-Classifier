# Fake News Classifier

A machine learning project that classifies news articles as either "true" or "fake" using various models, from simple baselines to advanced neural networks.

## Project Overview

This project implements and compares different approaches to fake news detection:
- Majority Class Classifier (Baseline)
- Logistic Regression with TF-IDF
- Fully Connected Neural Network
- Advanced LSTM Model with Attention Mechanism

The best performing model (Advanced LSTM) achieved:
- Accuracy: 96.19%
- Precision: 94.88%
- Recall: 98.33%
- F1-Score: 96.58%

## Documentation

- [Technical Report](docs/fake_news_classifier.pdf) - Detailed description of the models, methodology, and results
- [Project Presentation](docs/presentation.pdf) - Visual overview of the project's key components and findings

## Key Features

- Text preprocessing with NLTK
- Pre-trained Word2Vec embeddings
- Bidirectional LSTM with attention mechanism
- Comprehensive model evaluation metrics
- Training visualization tools

## Model Architecture

The advanced model includes:
- Embedding Layer (Word2Vec)
- Bidirectional LSTM
- Attention Mechanism
- Dropout Regularization
- Dense Layers

## Dataset

The project uses two datasets:
1. "True and Fake News" dataset by Sameer Patel
2. "WELFake" dataset by Saurabh Shahane

## Dependencies

- PyTorch
- NLTK
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Gensim (for Word2Vec)

## Authors

- Yocheved Ofstein
- Shay Gali
- Shalom Ofstein
