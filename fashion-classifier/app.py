"""
app.py — FastAPI entry point for the Fashion Classifier service.

Render start command:
    uvicorn app:app --host 0.0.0.0 --port $PORT

Working directory on Render:
    fashion-classifier/fashion-classifier/   (rootDir setting)

Model:
    public/model/model.json  (TF.js layers-model, MobileNetV2-based)
    public/model/group1-shard1of1.bin

Input contract (mirrors UploadSwatch.tsx + lib/model.ts in the Next.js app):
    • Resize image to 96×96 with letterbox (mid-grey #808080 background).
    • Normalize each channel to [-1, 1] via  value / 127.5 - 1.0
    • Feed as float32 tensor shape [1, 96, 96, 3].

Output:
    Softmax probabilities over 10 Fashion-MNIST-compatible classes.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("fashion_classifier")

# ---------------------------------------------------------------------------
# Constants — must match training config and UploadSwatch.tsx
# ---------------------------------------------------------------------------
IMG_SIZE = 96          # model input resolution (pixels)
NUM_CLASSES = 10

CLASS_NAMES: List[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Path to TF.js model directory, relative to this file's location.
# On Render the CWD is fashion-classifier/fashion-classifier/ (rootDir),
# which is also where app.py lives — so the relative path below is correct.
_HERE = Path(__file__).parent
MODEL_DIR = _HERE / "public" / "model"
MODEL_JSON = MODEL_DIR / "model.json"

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------
_keras_model = None  # populated by _load_model()


def _load_model():
    """
    Load the TF.js layers-model from public/model/model.json and convert it
    to a Keras model in memory using tensorflowjs.converters.

    The model weights live in group1-shard1of1.bin alongside model.json;
    tensorflowjs resolves them automatically when given the directory path.
    """
    global _keras_model
    if _keras_model is not None:
        return _keras_model

    if not MODEL_JSON.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_JSON}\n"
            "Run 'npm run stub:model' or 'npm run train:model' to generate "
            "public/model/model.json before starting the server."
        )

    logger.info("Loading TF.js model from %s …", MODEL_JSON)
    try:
        import tensorflowjs as tfjs  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "tensorflowjs is not installed. "
            "Run: pip install -r scripts/requirements.txt"
        ) from exc
    try:
        _keras_model = tfjs.converters.load_keras_model(str(MODEL_JSON))
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc

    logger.info("Model loaded successfully. Input shape: %s", _keras_model.input_shape)
    return _keras_model


# ---------------------------------------------------------------------------
# Preprocessing — mirrors UploadSwatch.tsx exactly
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes into a float32 array of shape [1, 96, 96, 3]
    ready for model inference.

    Steps:
      1. Decode image (any format Pillow supports) as RGB.
      2. Letterbox-resize to IMG_SIZE × IMG_SIZE using mid-grey (#808080)
         background — same as the canvas fillStyle in UploadSwatch.tsx.
      3. Normalize: value / 127.5 - 1.0  →  range [-1, 1].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Letterbox: fit the image inside a square canvas preserving aspect ratio.
    orig_w, orig_h = img.size
    scale = IMG_SIZE / max(orig_w, orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Mid-grey canvas (128 = 0x80), then paste centred.
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
    offset_x = (IMG_SIZE - new_w) // 2
    offset_y = (IMG_SIZE - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    # HWC uint8 → float32 [-1, 1], then add batch dimension → [1, 96, 96, 3]
    arr = np.array(canvas, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return arr[np.newaxis, ...]   # shape [1, 96, 96, 3]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Pre-load the Keras model before the server accepts requests."""
    try:
        _load_model()
    except RuntimeError as exc:
        # Log but don't crash — /predict will surface the error gracefully.
        logger.warning("Model could not be loaded at startup: %s", exc)
    yield
    # (shutdown logic could go here if needed)


app = FastAPI(
    title="Fashion Classifier API",
    description=(
        "MobileNetV2-based garment classifier. "
        "Upload a clothing photo to GET a prediction over 10 Fashion-MNIST classes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Next.js frontend (any origin in dev; restrict in production as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    message: str


class Prediction(BaseModel):
    class_index: int
    class_name: str
    confidence: float


class PredictResponse(BaseModel):
    top: Prediction
    probabilities: List[Prediction]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["health"])
def health_check():
    """Returns a simple JSON confirming the API is running."""
    return HealthResponse(
        status="ok",
        message="Fashion Classifier API is running. POST an image to /predict.",
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    """
    Classify a garment image.

    - **file**: An image file (JPEG, PNG, WebP, etc.)

    Returns the top prediction and the full probability distribution over
    all 10 classes.
    """
    # ── Validate content type ────────────────────────────────────────────────
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded file must be an image, got: {file.content_type}",
        )

    # ── Read and preprocess ──────────────────────────────────────────────────
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        logger.error("Image preprocessing failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode image: {exc}",
        )

    # ── Inference ────────────────────────────────────────────────────────────
    try:
        model = _load_model()
        probs = model.predict(tensor, verbose=0)[0]   # shape [10]
    except RuntimeError as exc:
        logger.error("Model error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # ── Build response ───────────────────────────────────────────────────────
    predictions = [
        Prediction(
            class_index=i,
            class_name=CLASS_NAMES[i],
            confidence=float(probs[i]),
        )
        for i in range(NUM_CLASSES)
    ]
    top_idx = int(np.argmax(probs))

    return PredictResponse(
        top=predictions[top_idx],
        probabilities=predictions,
    )


