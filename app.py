
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Customer Intent Predictor",
    page_icon="💬",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "intent_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("💬 Customer Support Intent Predictor")
st.write(
    "Enter a customer message below to predict its support intent."
)

try:
    model = load_model()
except Exception:
    st.error(
        "Could not load the model. "
        "Check that intent_model.pkl is in the repository root."
    )
    st.stop()

message = st.text_area(
    "Customer message",
    placeholder="Example: I want to cancel my order."
)

if st.button("Predict Intent", type="primary"):
    if not message.strip():
        st.warning("Please enter a customer message.")
    else:
        prediction = model.predict([message.strip()])[0]
        st.success(f"Predicted Intent: **{prediction}**")
