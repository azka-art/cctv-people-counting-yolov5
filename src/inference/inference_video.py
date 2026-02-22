"""
Video inference pipeline for People Detection & Counting.

Uses:
- OpenCV VideoCapture/VideoWriter for video I/O
- YOLOv5s via torch.hub for per-frame person detection

Supports --enhance flag for CLAHE + tile-based inference.

Usage:
    # Standard
    python -m src.inference.inference_video \
        --input assets/demo_input.mp4 \
        --output assets/demo_output.mp4 \
        --conf 0.4 --device cpu

    # Enhanced (better accuracy, slower)
    python -m src.inference.inference_video \
        --input assets/demo_input.mp4 \
        --output assets/demo_output.mp4 \
        --conf 0.25 --device cpu --enhance
"""

import argparse
import sys
from pathlib import Path

import cv2

from src.inference.inference_image import annotate_image, detect_people, load_model


def run_video_inference(
    input_path: str,
    output_path: str,
    conf: float = 0.4,
    device: str = "cpu",
    enhance: bool = False,
):
    """
    Full video inference pipeline.

    1. Open video with OpenCV VideoCapture
    2. Run YOLOv5s detection per frame (standard or enhanced)
    3. Annotate each frame with bbox + confidence + People Count
    4. Write output video with matching FPS and resolution
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    if enhance:
        print("[INFO] Enhanced mode: CLAHE + tile-based inference enabled")

    # Setup output writer
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[ERROR] Cannot create output video writer")
        sys.exit(1)

    # Load model once
    model = load_model(device)

    # Import enhancements if needed
    if enhance:
        from src.inference.enhance import apply_clahe, tile_inference

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if enhance:
            # Apply CLAHE preprocessing
            enhanced_bgr = apply_clahe(frame_bgr)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

            # Tile-based inference on enhanced frame
            detections = tile_inference(model, enhanced_rgb, conf=conf)
            count = len(detections)
        else:
            # Standard inference
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections, count = detect_people(model, frame_rgb, conf)

        # Annotate on original frame (not CLAHE) for natural output
        annotated = annotate_image(frame_bgr, detections, count)

        # Write frame
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[INFO] Processed {frame_idx}/{total_frames} frames - {count} people detected")

    cap.release()
    writer.release()

    print(f"[INFO] Done! Processed {frame_idx} frames")
    print(f"[INFO] Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="People Detection - Video Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output annotated video")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--enhance", action="store_true",
                        help="Enable CLAHE + tile-based inference for better accuracy")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    run_video_inference(args.input, args.output, args.conf, args.device, args.enhance)


if __name__ == "__main__":
    main()