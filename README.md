# FinalProject-Group_AGandJM
NLP Final project
Project Overview

This project explores how Natural Language Processing (NLP) techniques can automatically classify financial news headlines by sentiment: positive, neutral, or negative.Financial headlines drive market behavior, and capturing their sentiment at scale helps analysts understand trends and investor reactions.

This work compares:

A classical NLP pipeline using TF-IDF + Logistic Regression
A Transformer-based model using fine-tuned DistilBERT
The final model is deployed using Streamlit to allow real-time sentiment prediction.
Objective:
Build and evaluate multiple NLP models for financial sentiment classification
Understand differences between classical and contextual NLP methods
Analyze model performance using metrics, confusion matrices, and embeddings
Deploy the best-performing model for real-time use

Dataset Description:
The dataset contains financial news headlines labeled as positive, neutral, or negative.
Sentiment distribution is imbalanced, with fewer positive examples.
Headlines are short, domain-specific, and often ambiguous — making the task challenging.
Preprocessing (cleaning, normalization) was performed collaboratively.
Preprocessing Summary

Basic text cleaning and normalization
Label encoding
Train/validation/test split
For Transformers:
Tokenization
Padding/truncation
Attention masks
Custom PyTorch Dataset class

Model Architecture
<img width="898" height="569" alt="image" src="https://github.com/user-attachments/assets/1807d86b-ce27-40b8-841a-419c456495df" />

The system contains two modeling branches:
Classical Baseline
TF-IDF vectorization (1–2 n-grams)
Logistic Regression with balanced class weights

Transformer Model
DistilBERT Encoder → Dropout → Linear Classifier → Softmax
Fine-tuned on labeled financial headlines
Optimizer: AdamW
LR: 2e-5
Epochs: 6
Batch size: 32
LR scheduler with warm-up
A Streamlit front-end loads the saved model for real-time inference.

Results
Baseline (TF-IDF + Logistic Regression)
Accuracy: 0.80
Macro-F1: 0.746
Strong on negative and neutral classes
Weak on positive sentiment due to subtlety + class imbalance
DistilBERT Transformer Model
Accuracy: 0.88
Macro-F1: 0.84
Learned deeper contextual patterns
Reduced errors across all sentiment categories

Confusion Matrix Analysis
<img width="521" height="409" alt="image" src="https://github.com/user-attachments/assets/2e34b47e-0fe6-432f-90f2-2e1bccb56ab1" />

Negative and neutral classes show high accuracy
Positive sentiment remains most challenging

Common errors:
Neutral → Negative (market terms interpreted as negative)
Negative → Neutral (sentiment implied indirectly)
Positive → Negative (optimism often subtle in finance)

Streamlit Deployment
A simple Streamlit interface allows users to:
Type any financial headline
Receive real-time sentiment prediction
See probability scores for each class
Model + tokenizer loaded using save_pretrained()
<img width="940" height="421" alt="image" src="https://github.com/user-attachments/assets/0b676df4-bbc3-4128-90ba-d27690a201e1" />
<img width="940" height="476" alt="image" src="https://github.com/user-attachments/assets/79ee52e7-ba23-4e2d-a37f-1c59d7a07c08" />

The fine-tuned DistilBERT model offered a substantial improvement over the TF-IDF baseline, largely because it captures the contextual cues that define sentiment in financial language. While negative and neutral headlines were modeled well, positive sentiment remained the most difficult category due to its subtle and infrequent expression. The embedding analysis confirmed that the Transformer learned meaningful structure in the headline space, and the Streamlit deployment showed how the model can be used interactively for real-time sentiment exploration.


