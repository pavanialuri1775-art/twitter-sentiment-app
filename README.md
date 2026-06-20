# Twitter Sentiment Analysis using Machine Learning

## Project Overview
This project performs sentiment analysis on Twitter data using Natural Language Processing (NLP) and Machine Learning.

The model classifies tweets into:
- Positive Sentiment
- Negative Sentiment

The project uses:
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes

---

# Features
- Data preprocessing and cleaning
- Removal of URLs, mentions, and special characters
- TF-IDF text vectorization
- Sentiment classification using Naive Bayes
- Accuracy evaluation
- Confusion Matrix and Classification Report
- Custom tweet prediction

---

# Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Regular Expressions (re)

---

# Dataset
The dataset contains Twitter tweets with sentiment labels.

### Labels
| Label | Meaning |
|-------|---------|
| 0 | Negative |
| 4 | Positive |

The labels are converted into:
- 0 → Negative
- 1 → Positive

---

# Project Workflow

## 1. Dataset Handling
- Load dataset using Pandas
- Select required columns
- Rename columns

## 2. Data Cleaning
The tweets are cleaned by:
- converting text to lowercase
- removing URLs
- removing usernames
- removing special characters

## 3. Train-Test Split
The dataset is divided into:
- 80% Training Data
- 20% Testing Data

## 4. TF-IDF Vectorization
Text data is converted into numerical vectors using TF-IDF.

## 5. Model Training
A Multinomial Naive Bayes model is trained on the dataset.

## 6. Model Evaluation
The model is evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

---

# Installation

## Clone Repository
```bash
git clone https://github.com/pavanialuri1775-art/twitter-sentiment-app.git
