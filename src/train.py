
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset CSV files
train = pd.read_csv(
    "data/archive/Bitext_Sample_Customer_Service_Training_Dataset.csv"
)
test = pd.read_csv(
    "data/archive/Bitext_Sample_Customer_Service_Testing_Dataset.csv"
)
val = pd.read_csv(
    "data/archive/Bitext_Sample_Customer_Service_Validation_Dataset.csv"
)

# Combine the datasets, following the original notebook
df = pd.concat([train, test, val], ignore_index=True)

# Clean the data
df = df.dropna(subset=["utterance", "intent"])
df = df.drop_duplicates()

X = df["utterance"].astype(str)
y = df["intent"]

# Split data for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, "intent_model.pkl")

print("\nModel saved as intent_model.pkl")
