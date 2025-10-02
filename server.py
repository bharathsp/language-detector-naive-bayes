from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# ------------------------------
# Load trained pipeline and label encoder
# ------------------------------
model = joblib.load("trained_pipeline-0.1.0.joblib")
label_encoder = joblib.load("label_encoder.joblib")  # load stored encoder

# ------------------------------
# Define FastAPI app
# ------------------------------
app = FastAPI(
    title="Language Detection API",
    description="An API to detect language from text using MultinomialNB + TF-IDF",
    version="0.1.0"
)

# ✅ Add CORS middleware here
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # In production, replace with your frontend domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Serve static HTML from /static
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ------------------------------
# Request body schema
# ------------------------------
class TextInput(BaseModel):
    text: str

# ------------------------------
# Root endpoint
# ------------------------------
# @app.get("/")
# def home():
#     return {"message": "Welcome to the Language Detection API 🚀"}

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
