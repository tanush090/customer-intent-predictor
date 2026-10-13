
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the dataset
train = pd.read_csv(
    "data/archive/Bitext_Sample_Customer_Service_Training_Dataset.csv"
)
test = pd.read_csv(
    "data/archive/Bitext_Sample_Customer_Service_Testing_Dataset.csv"
)
val = pd.read_csv(
    "data/archive/Bitext_Sample_Customer_Service_Validation_Dataset.csv"
)

df = pd.concat([train, test, val], ignore_index=True)
df = df.dropna(subset=["utterance", "intent"]).drop_duplicates()

X = df["utterance"].astype(str)
y = df["intent"]

# Recreate the same split used during training
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load the trained pipeline
model = joblib.load("intent_model.pkl")

# Evaluate
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
