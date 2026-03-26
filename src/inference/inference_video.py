"""
Video inference pipeline for People Detection & Counting.

Uses:
- OpenCV VideoCapture/VideoWriter for video I/O
- YOLOv5s via torch.hub for per-frame person detection
- SORTTracker (Kalman filter) for persistent person IDs

Modes:
    Standard:  per-frame detection count, no tracking
    Enhanced:  CLAHE + tile-based inference for crowded/backlit scenes (--enhance)
    Tracking:  persistent person IDs + ID switch counting (--track)

CLI Usage:
    # Standard
    python -m src.inference.inference_video \\
        --input  assets/demo_input.mp4 \\
        --output assets/demo_output.mp4 \\
        --conf 0.4 --device cpu

    # Enhanced + Tracking (recommended for crowded BRT stations)
    python -m src.inference.inference_video \\
        --input  assets/demo_input.mp4 \\
        --output assets/demo_output.mp4 \\
        --conf 0.3 --device cpu --enhance --track

    # Tracking with custom Kalman age tolerance
    python -m src.inference.inference_video \\
        --input  assets/demo_input.mp4 \\
        --output assets/demo_output.mp4 \\
        --conf 0.3 --device cpu --track --max-age 45

Expected output:
    - Annotated video at original FPS + resolution
    - Bounding boxes with confidence scores per detection
    - Top-left overlay: People Count / Unique Total / ID Switches
    - Terminal summary: per-frame progress + final tracking metrics
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.inference.inference_image import annotate_image, detect_people, load_model


# ---------------------------------------------------------------------------
# Color palette — 20 visually distinct BGR colors for track IDs
# ---------------------------------------------------------------------------
TRACK_COLORS = [
    (0,   255,   0),   # green
    (255,   0,   0),   # blue
    (0,     0, 255),   # red
    (255, 255,   0),   # cyan
    (0,   255, 255),   # yellow
    (255,   0, 255),   # magenta
    (128, 255,   0),   # lime
    (0,   128, 255),   # orange
    (255, 128,   0),   # sky blue
    (128,   0, 255),   # purple
    (0,   200, 200),   # teal
    (200,   0, 200),   # pink
    (200, 200,   0),   # olive
    (0,   128, 128),   # dark teal
    (128, 128,   0),   # dark olive
    (64,  255, 128),   # mint
    (255,  64, 128),   # coral
    (128,  64, 255),   # lavender
    (255, 200,   0),   # gold
    (0,    64, 200),   # navy
]


# ---------------------------------------------------------------------------
# Frame annotation helpers
# ---------------------------------------------------------------------------

def _draw_label(
    img: np.ndarray,
    text: str,
    origin: tuple,
    color: tuple,
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw a filled-background label at origin (x, y — bottom-left of text)."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = origin
    cv2.rectangle(
        img,
        (x, y - th - baseline - 4),
        (x + tw + 2, y + 2),
        color, cv2.FILLED,
    )
    # Black text on colored background
    cv2.putText(
        img, text, (x + 1, y - baseline - 1),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
    )


def annotate_with_tracks(
    image_bgr: np.ndarray,
    tracked_dets: list,
    frame_count: int,
    unique_count: int,
    id_switches: int,
) -> np.ndarray:
    """
    Annotate frame with track-colored bounding boxes and metrics overlay.

    Draws:
      - Color-coded bounding box per tracked person (color = track ID)
      - Label: "ID:N  0.87" above each box
      - Top-left overlay block:
            People Count : N   (active detections this frame)
            Unique Total : N   (unique IDs assigned so far)
            ID Switches  : N   (tracking quality indicator)

    Args:
        image_bgr:    BGR frame from OpenCV
        tracked_dets: List of dicts {x1, y1, x2, y2, score, track_id}
        frame_count:  Active detections in current frame
        unique_count: Total unique person IDs assigned since video start
        id_switches:  Total ID switches detected since video start

    Returns:
        Annotated BGR frame (copy, original unmodified)
    """
    annotated = image_bgr.copy()

    # --- Per-person bounding boxes ---
    for det in tracked_dets:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        score    = det["score"]
        track_id = det["track_id"]

        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label above box
        _draw_label(
            annotated,
            f"ID:{track_id}  {score:.2f}",
            origin=(x1, y1),
            color=color,
            font_scale=0.48,
        )

    # --- Top-left metrics overlay ---
    overlay_lines = [
        (f"People Count : {frame_count}",  (0, 220,   0)),   # bright green
        (f"Unique Total : {unique_count}", (0, 200, 255)),   # yellow-ish
        (f"ID Switches  : {id_switches}",  (0, 128, 255)),   # orange
    ]

    y_cursor = 12
    for text, color in overlay_lines:
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
        )
        # Semi-transparent black background
        cv2.rectangle(
            annotated,
            (8,  y_cursor - 4),
            (18 + tw, y_cursor + th + 6),
            (0, 0, 0), cv2.FILLED,
        )
        cv2.putText(
            annotated, text, (13, y_cursor + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2,
        )
        y_cursor += th + 14

    return annotated


# ---------------------------------------------------------------------------
# Core inference pipeline
# ---------------------------------------------------------------------------

def run_video_inference(
    input_path: str,
    output_path: str,
    conf: float = 0.4,
    device: str = "cpu",
    enhance: bool = False,
    track: bool = False,
    max_age: int = 30,
) -> dict:
    """
    Full video inference pipeline.

    Steps:
        1. Open video with cv2.VideoCapture
        2. Load YOLOv5s model once (weights auto-downloaded to ~/.cache)
        3. Per frame: detect → (optionally) enhance → (optionally) track → annotate
        4. Write annotated frames to output MP4 (codec: mp4v)
        5. Print + return final metrics summary

    Args:
        input_path:  Path to source video file
        output_path: Path for annotated output video
        conf:        YOLOv5 confidence threshold (default: 0.4)
        device:      "cpu" or "cuda"
        enhance:     Enable CLAHE + tile-based inference
        track:       Enable SORTTracker for persistent IDs
        max_age:     Kalman tracker max frames without detection before
                     track deletion (default: 30 ≈ 1 s at 30 fps)

    Returns:
        dict with keys: total_frames, total_unique, id_switches,
                        active_tracks, avg_fps_processing
    """
    # ----------------------------------------------------------------
    # Open input
    # ----------------------------------------------------------------
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input  : {input_path}")
    print(f"[INFO] Video  : {width}×{height} @ {fps:.1f} FPS  ({total_frames} frames)")
    print(f"[INFO] Conf   : {conf}  |  Device : {device}")
    print(f"[INFO] Enhance: {'ON  (CLAHE + tile inference)' if enhance else 'OFF'}")
    print(f"[INFO] Track  : {'ON  (SORT + Kalman, max_age=' + str(max_age) + ')' if track else 'OFF'}")
    print()

    # ----------------------------------------------------------------
    # Setup output writer
    # ----------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[ERROR] Cannot create output video writer. "
              "Check that the output directory exists and is writable.")
        cap.release()
        sys.exit(1)

    # ----------------------------------------------------------------
    # Load model + optional modules
    # ----------------------------------------------------------------
    model = load_model(device)

    if enhance:
        from src.inference.enhance import apply_clahe, tile_inference

    tracker = None
    if track:
        from src.inference.tracker import SORTTracker
        tracker = SORTTracker(
            max_age=max_age,
            min_hits=1,
            iou_threshold=0.3,
        )

    # ----------------------------------------------------------------
    # Frame loop
    # ----------------------------------------------------------------
    import time
    frame_idx      = 0
    total_detected = 0
    t_start        = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # --- Detection ---
        if enhance:
            from src.inference.enhance import apply_clahe, tile_inference
            enhanced_bgr = apply_clahe(frame_bgr)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            detections   = tile_inference(model, enhanced_rgb, conf=conf)
            count        = len(detections)
        else:
            frame_rgb          = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections, count  = detect_people(model, frame_rgb, conf)

        total_detected += count

        # --- Tracking / Annotation ---
        if track and tracker is not None:
            tracked    = tracker.update(detections)
            id_switches = tracker.get_id_switches()
            annotated  = annotate_with_tracks(
                frame_bgr,
                tracked,
                frame_count=len(tracked),
                unique_count=tracker.get_unique_count(),
                id_switches=id_switches,
            )
        else:
            annotated = annotate_image(frame_bgr, detections, count)

        writer.write(annotated)
        frame_idx += 1

        # --- Progress log every 30 frames ---
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            elapsed   = time.perf_counter() - t_start
            proc_fps  = frame_idx / elapsed if elapsed > 0 else 0
            track_str = ""
            if track and tracker:
                m = tracker.get_metrics()
                track_str = (
                    f" | Unique: {m['total_unique']}"
                    f"  IDSW: {m['id_switches']}"
                    f"  Active: {m['active_tracks']}"
                )
            print(
                f"[INFO] Frame {frame_idx:>4}/{total_frames}"
                f"  Detected: {count:>3}"
                f"  Proc: {proc_fps:.1f} fps"
                f"{track_str}"
            )

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    cap.release()
    writer.release()

    elapsed_total = time.perf_counter() - t_start
    avg_proc_fps  = frame_idx / elapsed_total if elapsed_total > 0 else 0

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    print("[DONE] Video inference complete")
    print(f"       Frames processed : {frame_idx}")
    print(f"       Processing speed : {avg_proc_fps:.2f} fps (avg)")
    print(f"       Output saved to  : {output_path}")

    metrics = {
        "total_frames":     frame_idx,
        "avg_fps_processing": round(avg_proc_fps, 2),
    }

    if track and tracker:
        m = tracker.get_metrics()
        print(f"       Total unique IDs : {m['total_unique']}")
        print(f"       ID switches      : {m['id_switches']}")
        print(f"       Active tracks    : {m['active_tracks']}")
        print(f"       Lost tracks      : {m['lost_tracks']}")
        print()
        print("  📊  Tracking Quality")
        if m["total_unique"] > 0:
            idsw_rate = m["id_switches"] / m["total_unique"] * 100
            print(f"       IDSW rate        : {idsw_rate:.1f}%  "
                  f"(ID switches / unique IDs)")
            if idsw_rate < 5:
                print("       Stability        : ✅ Good  (< 5%)")
            elif idsw_rate < 15:
                print("       Stability        : ⚠️  Fair  (5–15%)")
            else:
                print("       Stability        : ❌ Poor  (> 15%) — consider tuning --max-age")

        metrics.update({
            "total_unique":   m["total_unique"],
            "id_switches":    m["id_switches"],
            "active_tracks":  m["active_tracks"],
        })

    print("=" * 60)
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="People Detection & Counting — Video Inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input video file (e.g. assets/demo_input.mp4)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for annotated output video (e.g. assets/demo_output.mp4)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="YOLOv5 confidence threshold (default: 0.4)\n"
             "Use 0.3 with --enhance for crowded scenes",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--enhance", action="store_true",
        help="Enable CLAHE preprocessing + tile-based inference\n"
             "Improves recall in crowded/backlit scenes (slower: ~0.3 fps on CPU)",
    )
    parser.add_argument(
        "--track", action="store_true",
        help="Enable SORT tracking with Kalman filter\n"
             "Adds persistent IDs, unique count, and ID switch metrics",
    )
    parser.add_argument(
        "--max-age", type=int, default=30,
        help="Tracker max frames without detection before track deletion\n"
             "Higher = more tolerant of brief occlusion (default: 30)",
    )

    args = parser.parse_args()

    # Validate input path
    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    run_video_inference(
        input_path=args.input,
        output_path=args.output,
        conf=args.conf,
        device=args.device,
        enhance=args.enhance,
        track=args.track,
        max_age=args.max_age,
    )


if __name__ == "__main__":
    main()