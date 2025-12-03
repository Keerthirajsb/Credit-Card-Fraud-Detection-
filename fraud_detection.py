import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
csv_path = r'C:\Users\KEERTHIRAJ\OneDrive\Desktop\MLproject\archive (1)\creditcard.csv'
df = pd.read_csv(csv_path)

# Display basic info
print("Dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Class distribution:\n", df['Class'].value_counts())

# Split data into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Initialize and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
