
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

from google.colab import drive
drive.mount('/content/gdrive')

df = pd.read_csv('gdrive/MyDrive/emails.csv')

df.head()

print(f'original dataset shape : {df.shape}')
print(f'duplicates found : {df.duplicated().sum()}')

df.drop_duplicates(inplace=True)
print(f"Shape after cleaning: {df.shape}")

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam', 'Spam'],
            yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#Test with a custom email (Demo)
print("\n--- Live Test ---")
sample_emails = [
    "Subject: You have won a guaranteed prize! Click here now.",
    "Subject: Meeting reminder. Hey, can we reschedule our meeting to Tuesday?",
    "Subject: Today properties prices are going down."
]
sample_vec = vectorizer.transform(sample_emails)
predictions = model.predict(sample_vec)

for text, label in zip(sample_emails, predictions):
    result = "Spam" if label == 1 else "Not Spam"
    print(f"Email: '{text}' -> Prediction: {result}")

