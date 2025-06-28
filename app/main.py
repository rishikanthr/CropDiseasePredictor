from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import numpy as np
from PIL import Image
import io, os
from pathlib import Path

# ───────────── CONFIG ─────────────
MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).parent / "model" / "plant_disease.keras"))
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Load model once
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise Exception(f"❌ Failed to load model: {str(e)}")

# FastAPI app init
app = FastAPI(title="🌿 Plant Disease Inference API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image preprocessing
def preprocess(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    except Exception as e:
        raise HTTPException(400, detail=f"Invalid image file: {str(e)}")
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"📤 Received file: {file.filename}, Content-Type: {file.content_type}")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, detail="Uploaded file must be an image (JPEG/PNG).")

    img_bytes = await file.read()
    input_arr = preprocess(img_bytes)

    preds = model.predict(input_arr)[0]
    idx = int(np.argmax(preds))
    prob = round(float(preds[idx]), 4)

    return {
        "class": CLASS_NAMES[idx],
        "prob": prob
    }

@app.get("/")
def root():
    return {"status": "🌱 Plant Disease API is running!"}
