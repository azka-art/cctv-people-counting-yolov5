"""
Video inference pipeline for People Detection & Counting.

Uses:
- OpenCV VideoCapture/VideoWriter for video I/O
- YOLOv5s via torch.hub for per-frame person detection

Modes:
- Standard: per-frame detection count
- Enhanced: CLAHE + tile-based inference (--enhance)
- Tracking: persistent person IDs across frames (--track)

Usage:
    # Standard
    python -m src.inference.inference_video \
        --input assets/demo_input.mp4 \
        --output assets/demo_output.mp4 \
        --conf 0.4 --device cpu

    # Enhanced + Tracking (recommended)
    python -m src.inference.inference_video \
        --input assets/demo_input.mp4 \
        --output assets/demo_output.mp4 \
        --conf 0.3 --device cpu --enhance --track
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.inference.inference_image import annotate_image, detect_people, load_model


# Distinct colors for track IDs (BGR format)
TRACK_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 0),  # lime
    (0, 128, 255),  # orange
    (255, 128, 0),  # sky blue
    (128, 0, 255),  # purple
]


def annotate_with_tracks(image_bgr: np.ndarray, tracked_dets: list,
                         frame_count: int, unique_count: int) -> np.ndarray:
    """
    Draw bounding boxes with track IDs and unique person count.

    Each tracked person gets a unique color based on their ID.

    Args:
        image_bgr: Image in BGR format
        tracked_dets: List of dicts with x1,y1,x2,y2,score,track_id
        frame_count: Current per-frame detection count
        unique_count: Total unique persons seen so far

    Returns:
        Annotated image in BGR format
    """
    annotated = image_bgr.copy()

    for det in tracked_dets:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        score = det["score"]
        track_id = det["track_id"]

        # Color based on track ID
        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label with ID and confidence
        label = f"ID:{track_id} {score:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - baseline - 4),
            (x1 + label_size[0], y1),
            color, cv2.FILLED,
        )
        cv2.putText(
            annotated, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

    # Overlay: current count + unique count
    lines = [
        f"People Count: {frame_count}",
        f"Unique Total: {unique_count}",
    ]
    y_offset = 15
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(
            annotated, (10, y_offset - 5),
            (20 + tw, y_offset + th + 5), (0, 0, 0), cv2.FILLED,
        )
        cv2.putText(
            annotated, line, (15, y_offset + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        y_offset += th + 15

    return annotated


def run_video_inference(
    input_path: str,
    output_path: str,
    conf: float = 0.4,
    device: str = "cpu",
    enhance: bool = False,
    track: bool = False,
):
    """
    Full video inference pipeline.

    1. Open video with OpenCV VideoCapture
    2. Run YOLOv5s detection per frame (standard or enhanced)
    3. Optionally track people across frames with persistent IDs
    4. Annotate each frame with bbox + confidence + People Count
    5. Write output video with matching FPS and resolution
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
    if track:
        print("[INFO] Tracking mode: persistent person IDs enabled")

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

    # Initialize tracker if needed
    tracker = None
    if track:
        from src.inference.tracker import SimpleTracker
        tracker = SimpleTracker(max_disappeared=int(fps), iou_threshold=0.3)

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

        if track and tracker is not None:
            # Update tracker and annotate with IDs
            tracked = tracker.update(detections)
            annotated = annotate_with_tracks(
                frame_bgr, tracked,
                frame_count=len(tracked),
                unique_count=tracker.get_unique_count(),
            )
        else:
            # Standard annotation (no tracking)
            annotated = annotate_image(frame_bgr, detections, count)

        # Write frame
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 30 == 0:
            track_info = ""
            if track and tracker:
                track_info = f" | Unique: {tracker.get_unique_count()}"
            print(f"[INFO] Processed {frame_idx}/{total_frames} frames"
                  f" - {count} people detected{track_info}")

    cap.release()
    writer.release()

    print(f"\n[INFO] Done! Processed {frame_idx} frames")
    print(f"[INFO] Output saved to: {output_path}")
    if track and tracker:
        print(f"[INFO] Total unique persons tracked: {tracker.get_unique_count()}")


def main():
    parser = argparse.ArgumentParser(description="People Detection - Video Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output annotated video")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--enhance", action="store_true",
                        help="Enable CLAHE + tile-based inference for better accuracy")
    parser.add_argument("--track", action="store_true",
                        help="Enable IoU-based tracking for persistent person IDs")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    run_video_inference(
        args.input, args.output, args.conf, args.device,
        args.enhance, args.track,
    )


if __name__ == "__main__":
    main()