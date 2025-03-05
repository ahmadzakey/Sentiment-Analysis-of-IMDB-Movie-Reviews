# Sentiment Analysis of IMDB Movie Reviews

## Introduction
This project aims to analyze the sentiment of IMDB movie reviews using machine learning models. The primary models used are **Logistic Regression** and **Naïve Bayes** to classify sentiment as either **positive** or **negative**.

## Requirements
### Import Libraries
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import sklearn
import openai
```

## Load Dataset
The dataset contains **50,000** rows and **2** columns:
- **reviews**: Text of movie reviews
- **sentiment**: Sentiment label (**positive** or **negative**)

**Sentiment Distribution in the Dataset:**
```
positive    25000
negative    25000
Name: count, dtype: int64
```

## Data Cleaning
The first step in data processing is cleaning the data using the following techniques:
- **Tokenization**
- **Lowercasing**
- **Stopwords Removal**
- **Remove Punctuation**
- **Regular Expression Cleaning**
- **Lexicon Normalization**

**Note:** Spell checking was not performed due to the large dataset size, which caused performance issues on my computer.

Once this process is complete, each processed review is rejoined into a single string.

## Data Visualization
A **Word Cloud** was used to analyze common words in positive and negative sentiments. However, the results could be improved as some meaningless characters still appear.

## Converting Text into Numerical Format
The text data is converted into numerical form using **TF-IDF (Term Frequency - Inverse Document Frequency)**.

## Models and Results
### **1. Logistic Regression**
```
              precision    recall  f1-score   support

    negative       0.90      0.87      0.89      4961
    positive       0.88      0.90      0.89      5039

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

### **2. Naïve Bayes**
```
Naive Bayes Accuracy: 0.8549
              precision    recall  f1-score   support

    negative       0.86      0.85      0.85      4961
    positive       0.85      0.86      0.86      5039

    accuracy                           0.85     10000
   macro avg       0.85      0.85      0.85     10000
weighted avg       0.85      0.85      0.85     10000
```

## Conclusion
From the results above, it is clear that **Logistic Regression** performs better with **89% accuracy**, compared to **Naïve Bayes at only 85.49%**. This model can be further improved with additional data processing techniques and model parameter tuning.

---
🚀 **Future Work:**
- Implement **deep learning models** like **LSTM** to enhance accuracy
- Improve **Word Cloud** by refining text cleaning processes
- Explore **more NLP techniques** for better feature extraction
