# 📧 Spam Classifier

A simple machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)**.

---

## 🚀 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to detect spam messages.

It:
- Cleans text data
- Converts text into numbers
- Trains a model
- Predicts new messages

---

## 📂 Dataset


- SMS Spam Collection Dataset  
- Loaded from:
https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

- Contains:
  - `spam` → unwanted messages  
  - `ham` → normal messages  

---

## 🧹 Preprocessing

The text is cleaned using:

- Lowercasing  
- Removing punctuation  
- Removing extra spaces  

Example:

Original: "FREE money!!!"  
Cleaned: "free money"

---

## 🔢 Feature Extraction

- Uses **CountVectorizer**
- Converts text into numbers (Bag of Words)

---

## 🤖 Model

- **Multinomial Naive Bayes**

Why?
- Fast  
- Good for text classification  
- High accuracy  

---

## 📊 Accuracy
