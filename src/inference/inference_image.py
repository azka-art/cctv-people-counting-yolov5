"""
Image inference pipeline for People Detection & Counting.

Uses:
- PIL.Image.open() with RGB conversion (explicit Pillow usage - DoD requirement)
- YOLOv5s via torch.hub for person detection (pinned to v7.0 for reproducibility)
- OpenCV for annotation drawing

Enhancements (addressing error_analysis.md failure cases):
- CLAHE preprocessing for backlight/low-light scenes (Case 4)
- Tile-based inference for small/distant people (Case 1, 5)
- Lower confidence threshold option for dense crowds (Case 1)

Usage:
    # Standard inference
    python -m src.inference.inference_image \
        --input assets/sample.jpg \
        --output assets/sample_outputs/out.jpg \
        --conf 0.4 --device cpu

    # Enhanced inference (recommended for crowded scenes)
    python -m src.inference.inference_image \
        --input assets/sample.jpg \
        --output assets/sample_outputs/out_enhanced.jpg \
        --conf 0.25 --device cpu --enhance
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Pinned YOLOv5 version for reproducibility
YOLOV5_REPO = "ultralytics/yolov5"
YOLOV5_MODEL = "yolov5s"


def load_model(device: str = "cpu"):
    """
    Load YOLOv5s model via torch.hub.

    Model is pinned to ultralytics/yolov5 v7.0 for deterministic behavior.
    Weights are auto-downloaded to ~/.cache/torch/hub/ on first run.
    """
    model = torch.hub.load(YOLOV5_REPO, YOLOV5_MODEL, pretrained=True, _verbose=False)
    model.to(device)
    model.conf = 0.4  # default, overridden per call
    model.classes = [0]  # person class only
    return model


def detect_people(model, image_rgb: np.ndarray, conf: float = 0.4):
    """
    Run detection on an RGB numpy array.

    Returns:
        list[dict]: Each dict has keys x1, y1, x2, y2, score
        int: people count
    """
    model.conf = conf
    results = model(image_rgb)

    detections = []
    for *xyxy, confidence, cls in results.xyxy[0].cpu().numpy():
        if int(cls) == 0 and confidence >= conf:
            detections.append({
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3]),
                "score": round(float(confidence), 2),
            })

    return detections, len(detections)


def annotate_image(image_bgr: np.ndarray, detections: list, count: int) -> np.ndarray:
    """
    Draw bounding boxes, confidence scores, and People Count overlay.

    Args:
        image_bgr: Image in BGR format (OpenCV convention)
        detections: List of detection dicts with x1,y1,x2,y2,score
        count: Total people count

    Returns:
        Annotated image in BGR format
    """
    annotated = image_bgr.copy()

    # Draw bounding boxes and confidence scores
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        score = det["score"]

        # Bounding box (green)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Confidence label background
        label = f"{score:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - baseline - 4),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            annotated, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

    # People Count overlay (top-left corner)
    overlay_text = f"People Count: {count}"
    (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(annotated, (10, 10), (20 + tw, 20 + th + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        annotated, overlay_text, (15, 15 + th),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
    )

    return annotated


def run_image_inference(
    input_path: str,
    output_path: str,
    conf: float = 0.4,
    device: str = "cpu",
    enhance: bool = False,
):
    """
    Full image inference pipeline.

    1. Load image with PIL (explicit Pillow usage - DoD requirement)
    2. Optionally apply CLAHE preprocessing
    3. Run YOLOv5s detection (standard or tile-based)
    4. Annotate and save output

    Args:
        input_path: Path to input image
        output_path: Path to save annotated output
        conf: Confidence threshold (default 0.4, use 0.25 with --enhance)
        device: 'cpu' or 'cuda'
        enhance: Enable CLAHE + tile-based inference
    """
    # --- PIL.Image.open() with RGB conversion (Definition of Done requirement) ---
    pil_image = Image.open(input_path).convert("RGB")
    image_rgb = np.array(pil_image)

    # Load model
    model = load_model(device)

    if enhance:
        from src.inference.enhance import apply_clahe, tile_inference

        # Step 1: CLAHE preprocessing (mitigates Case 4 - backlight/low-light)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        enhanced_bgr = apply_clahe(image_bgr)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        print("[INFO] CLAHE preprocessing applied")

        # Step 2: Tile-based inference (mitigates Case 1, 5 - occlusion/distance)
        detections = tile_inference(model, enhanced_rgb, conf=conf)
        count = len(detections)
        print(f"[INFO] Tile-based inference: {count} people detected")

        # Annotate on original image (not CLAHE) for natural-looking output
        annotated = annotate_image(image_bgr, detections, count)
    else:
        # Standard inference
        detections, count = detect_people(model, image_rgb, conf)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        annotated = annotate_image(image_bgr, detections, count)

    # Save output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    print(f"[INFO] Detections: {count} people")
    print(f"[INFO] Output saved to: {output_path}")

    return detections, count


def main():
    parser = argparse.ArgumentParser(description="People Detection - Image Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to output annotated image")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--enhance", action="store_true",
                        help="Enable CLAHE + tile-based inference for better accuracy")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    run_image_inference(args.input, args.output, args.conf, args.device, args.enhance)


if __name__ == "__main__":
    main()