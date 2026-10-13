
import joblib
from pathlib import Path

# Locate the trained model in the project root
MODEL_PATH = Path(__file__).resolve().parent.parent / "intent_model.pkl"

def predict_intent(message):
    if not message or not message.strip():
        raise ValueError("Please enter a customer message.")

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([message.strip()])[0]
    return prediction

if __name__ == "__main__":
    print("Customer Support Intent Predictor")
    message = input("Enter a customer message: ")

    try:
        intent = predict_intent(message)
        print("Predicted Intent:", intent)
    except (ValueError, FileNotFoundError) as error:
        print("Error:", error)
