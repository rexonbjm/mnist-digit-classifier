import base64
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
MODEL_PATH = PROJECT_ROOT / "model" / "mnist_model.keras"

app = FastAPI(title="MNIST Digit Classifier")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
    print("Model summary:")
    model.summary()
except Exception as exc:
    print(f"Error loading model: {exc}")
    import traceback
    traceback.print_exc()
    model = None

# Static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionRequest(BaseModel):
    image: str  # base64 data URL


@app.get("/")
async def root_page():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend assets missing")
    return FileResponse(index_path)


@app.post("/predict")
async def predict(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Expect data URL like "data:image/png;base64,..."
        if "," in req.image:
            b64_data = req.image.split(",", 1)[1]
        else:
            b64_data = req.image
        image_bytes = base64.b64decode(b64_data)

        # Convert to grayscale and resize
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array
        arr = np.array(image).astype("float32")
        
        # CRITICAL: Invert colors if needed (MNIST expects white digit on black background)
        # User drawings are typically black on white, so we need to invert
        if arr.mean() > 127:
            arr = 255 - arr
        
        # Normalize to 0-1 range
        arr = arr / 255.0
        
        # Reshape to match training format: (1, 28, 28, 1)
        arr = arr.reshape(1, 28, 28, 1)

        preds = model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(preds))
        confidence = float(np.max(preds))
        probabilities = {str(i): float(preds[i]) for i in range(10)}

        return {"digit": digit, "confidence": confidence, "probabilities": probabilities}

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error processing image: {exc}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
