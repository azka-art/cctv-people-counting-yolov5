"""
FastAPI application for People Detection & Counting.

Endpoints:
    GET  /              -> Health check {"status": "ok"}
    POST /detect/image  -> Detect people in uploaded image

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

import io

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Query, UploadFile
from PIL import Image

from src.api.schemas import Detection, DetectionResponse, HealthResponse
from src.inference.enhance import apply_clahe, tile_inference
from src.inference.inference_image import detect_people, load_model

app = FastAPI(
    title="People Detection & Counting API",
    description="CCTV-like people detection using YOLOv5s for TransJakarta BRT monitoring",
    version="1.0.0",
)

# Model singleton
_model = None


def get_model():
    """Lazy-load the model singleton."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = load_model(device)
    return _model


@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup for faster first request."""
    print("[INFO] Loading YOLOv5s model...")
    get_model()
    print("[INFO] Model loaded successfully")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Docker verification."""
    return HealthResponse(status="ok")


@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    enhance: bool = Query(False, description="Enable CLAHE + tile-based inference"),
    conf: float = Query(0.4, description="Confidence threshold (use 0.25 with enhance=true)"),
):
    """
    Detect people in an uploaded image.

    - **file**: Image file (JPEG, PNG, etc.)
    - **enhance**: Enable CLAHE preprocessing + tile-based inference (default: false)
    - **conf**: Confidence threshold, 0.0-1.0 (default: 0.4, recommended 0.25 with enhance)
    - Returns: count + list of detections with bounding boxes and confidence scores
    """
    # Read uploaded file
    contents = await file.read()

    # Open with PIL and convert to RGB (explicit Pillow usage)
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_rgb = np.array(pil_image)

    model = get_model()

    if enhance:
        # CLAHE + tile-based inference
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        enhanced_bgr = apply_clahe(image_bgr)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        detections = tile_inference(model, enhanced_rgb, conf=conf)
        count = len(detections)
    else:
        # Standard inference
        detections, count = detect_people(model, image_rgb, conf=conf)

    return DetectionResponse(
        count=count,
        detections=[Detection(**d) for d in detections],
    )