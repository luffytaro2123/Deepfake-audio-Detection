# 🎙️ Deepfake Audio Detection using Machine Learning

A machine learning-based system for detecting **fake (synthetically generated) audio** using **MFCC feature extraction** and multiple classification models. The project preprocesses audio samples, extracts acoustic features, trains different ML models, and evaluates their performance across multiple datasets.

---

## 📌 Project Overview

Deepfake audio generated through AI voice synthesis can be misused for impersonation, fraud, and misinformation. This project aims to identify whether an audio sample is **real or fake** by analyzing speech characteristics using machine learning techniques.

The workflow includes:

- Audio preprocessing and cleaning
- Feature extraction using MFCCs
- Model training and evaluation
- Comparative analysis across datasets

---

## 🚀 Features

✅ Removes invalid / corrupted audio files (0-byte files)  
✅ Detects and removes duplicate audio samples using MD5 hashing  
✅ Audio loading and zero-padding to fixed length (16 kHz)  
✅ MFCC-based feature extraction  
✅ Feature normalization using StandardScaler  
✅ Multiple ML models for comparison  
✅ Performance evaluation using Accuracy, Precision, Recall, F1-score  
✅ Confusion Matrix and ROC curve generation  
✅ Model export using Joblib  

---

## 🛠 Tech Stack

- **Python**
- NumPy
- Pandas
- Librosa
- Scikit-learn
- XGBoost
- Matplotlib
- Joblib
- TQDM

---

## 📂 Dataset Structure

Project supports evaluation on multiple datasets:

```text
Dataset/
│── for-2seconds/
│── for-norm/
│── for-original/
└── for-rerecorded/
