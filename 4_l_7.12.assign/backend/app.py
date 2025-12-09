from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import cv2
from preprocess import load_and_preprocess

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# Model paths
MODEL_PATHS = {
    "logreg": "models/logreg_model.joblib",
    "knn": "models/knn_model.joblib",
    "gnb": "models/gnb_model.joblib"
}

# Shared scaler + encoder
SCALER_PATH = "models/scaler.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

# Load scaler + encoder once
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# Preload models once (Faster than loading every request)
MODELS = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


@app.get("/")
def home():
    return {"message": "API running", "models": list(MODELS.keys())}


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    file: UploadFile = File(...)
):
    if model_name not in MODELS:
        return {"error": f"Invalid model name '{model_name}'"}

    model = MODELS[model_name]

    # Read uploaded file memory → numpy array
    image_bytes = await file.read()
    img_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid or unreadable image"}

    # Preprocess image using your defined function
    features = load_and_preprocess(img).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    pred_label = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(probabilities))

    decoded_label = encoder.inverse_transform([pred_label])[0]

    return {
        "model_used": model_name,
        "brand": decoded_label,
        "confidence": confidence
    }
