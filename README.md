# 📧 Email Spam Detection System

A Machine Learning project that classifies emails as "Spam" or "Not Spam" (Ham) using Natural Language Processing (NLP) and the Naive Bayes algorithm.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Project Overview
Spam detection is a classic binary classification problem in Natural Language Processing. This project builds a model to automatically filter unwanted emails. 

By analyzing a dataset of over 5,000 emails, the model identifies patterns in text (such as frequent keywords used in spam) to predict whether an incoming email is malicious or legitimate.

### Key Results
- **Accuracy:** 99.2%
- **False Positives:** Extremely low (~3 emails misclassified as spam out of 1,139 in the test set).
- **Model:** Multinomial Naive Bayes.

## 🛠️ Technologies Used
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and cleaning.
- **Scikit-Learn**: Machine learning (CountVectorizer, Naive Bayes, Metrics).
- **Matplotlib & Seaborn**: Data visualization (Confusion Matrix).

## ⚙️ How It Works
1. **Data Preprocessing**: 
   - Loaded the `emails.csv` dataset.
   - Removed duplicate entries to ensure data quality.
2. **Feature Extraction**: 
   - Used `CountVectorizer` to convert text data into a matrix of token counts (Bag of Words model).
3. **Model Training**: 
   - Split data into training (80%) and testing (20%) sets.
   - Trained a `MultinomialNB` classifier, which is highly effective for discrete text counts.
4. **Evaluation**: 
   - Measured performance using Accuracy Score, Confusion Matrix, and Classification Report.

## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/email-spam-detector.git](https://github.com/yourusername/email-spam-detector.git)
   cd email-spam-detector
