from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# ------------------------------
# Load trained pipeline
# ------------------------------
model = joblib.load("trained_pipeline-0.1.0.joblib")

# ------------------------------
# Define FastAPI app
# ------------------------------
app = FastAPI(
    title="Language Detection API",
    description="An API to detect language from text using MultinomialNB + TF-IDF",
    version="0.1.0"
)

# ------------------------------
# Request body schema
# ------------------------------
class TextInput(BaseModel):
    text: str

# ------------------------------
# Root endpoint
# ------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Language Detection API 🚀"}

# ------------------------------
# Prediction endpoint
# ------------------------------
@app.post("/predict")
def predict_language(input: TextInput):
    """Predict the language of given text."""
    pred_encoded = model.predict([input.text])[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]  # decode back
    return {
        "text": input.text,
        "predicted_language": pred_label
    }
