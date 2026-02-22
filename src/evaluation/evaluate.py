"""
Evaluation script for People Detection & Counting.

Compares YOLOv5s detection counts against MOT20 ground truth annotations
and computes MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error).

Supports --enhance flag for CLAHE + tile-based inference evaluation.

Usage:
    # Standard evaluation
    python -m src.evaluation.evaluate \
        --dataset data/mot20/train/MOT20-01 \
        --conf 0.4 --device cpu \
        --output assets/sample_outputs/eval_results.json \
        --save-samples assets/sample_outputs/ --max-frames 50

    # Enhanced evaluation
    python -m src.evaluation.evaluate \
        --dataset data/mot20/train/MOT20-01 \
        --conf 0.3 --device cpu --enhance \
        --output assets/sample_outputs/eval_results_enhanced.json \
        --save-samples assets/sample_outputs/ --max-frames 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.inference.inference_image import annotate_image, detect_people, load_model


def load_mot_ground_truth(gt_path: str) -> dict:
    """
    Load MOT format ground truth and count unique persons per frame.

    MOT format: frame, id, x, y, w, h, conf, cls, vis
    - conf == 0 means 'ignore' in MOT20
    - cls == 1 means 'pedestrian', cls == 2 means 'person on vehicle'

    Returns:
        dict mapping frame_number -> person_count
    """
    gt_counts = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            frame_id = int(parts[0])
            conf = int(parts[6])
            cls = int(parts[7])

            # Only count valid pedestrian annotations
            if conf == 0:  # ignore region
                continue
            if cls not in (1, 2):  # 1=pedestrian, 2=person on vehicle
                continue

            gt_counts[frame_id] = gt_counts.get(frame_id, 0) + 1

    return gt_counts


def evaluate(
    dataset_path: str,
    conf: float = 0.4,
    device: str = "cpu",
    output_path: str = None,
    save_samples_dir: str = None,
    max_frames: int = None,
    enhance: bool = False,
):
    """
    Run evaluation on a MOT20 sequence.

    Args:
        dataset_path: Path to MOT20 sequence (e.g., data/mot20/train/MOT20-01)
        conf: Confidence threshold
        device: cpu or cuda
        output_path: Path to save JSON results
        save_samples_dir: Directory to save frames with high error
        max_frames: Limit number of frames (for quick testing)
        enhance: Enable CLAHE + tile-based inference
    """
    dataset_path = Path(dataset_path)
    img_dir = dataset_path / "img1"
    gt_path = dataset_path / "gt" / "gt.txt"

    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        sys.exit(1)

    # Load ground truth
    gt_counts = load_mot_ground_truth(str(gt_path))
    print(f"[INFO] Ground truth loaded: {len(gt_counts)} frames with annotations")

    # Get sorted image files
    image_files = sorted(img_dir.glob("*.jpg"))
    if max_frames:
        image_files = image_files[:max_frames]
    print(f"[INFO] Processing {len(image_files)} frames")

    mode_label = "Enhanced (CLAHE + tile)" if enhance else "Standard"
    print(f"[INFO] Mode: {mode_label} | Confidence: {conf}")

    # Load model
    model = load_model(device)

    # Import enhancements if needed
    if enhance:
        from src.inference.enhance import apply_clahe, tile_inference

    # Evaluate frame by frame
    results = []
    errors = []
    abs_errors = []
    total_inference_time = 0.0

    for img_path in image_files:
        frame_num = int(img_path.stem)
        gt_count = gt_counts.get(frame_num, 0)

        # Load frame
        frame_bgr = cv2.imread(str(img_path))

        start_time = time.time()

        if enhance:
            # CLAHE preprocessing + tile-based inference
            enhanced_bgr = apply_clahe(frame_bgr)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            detections = tile_inference(model, enhanced_rgb, conf=conf)
            pred_count = len(detections)
        else:
            # Standard inference
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections, pred_count = detect_people(model, frame_rgb, conf)

        inference_time = time.time() - start_time
        total_inference_time += inference_time

        error = pred_count - gt_count
        abs_error = abs(error)

        results.append({
            "frame": frame_num,
            "ground_truth": gt_count,
            "predicted": pred_count,
            "error": error,
            "abs_error": abs_error,
        })
        errors.append(error)
        abs_errors.append(abs_error)

        # Save frames with high error for error analysis
        if save_samples_dir and abs_error > 2:
            save_dir = Path(save_samples_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            annotated = annotate_image(frame_bgr, detections, pred_count)

            # Add ground truth info at bottom
            gt_text = f"GT: {gt_count} | Pred: {pred_count} | Err: {error:+d}"
            cv2.putText(
                annotated, gt_text, (15, frame_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            suffix = "_enhanced" if enhance else "_standard"
            save_path = save_dir / f"error_frame_{frame_num:06d}{suffix}.jpg"
            cv2.imwrite(str(save_path), annotated)

        # Progress log
        frame_idx = len(results)
        if frame_idx % 10 == 0:
            print(f"[INFO] Frame {frame_idx}/{len(image_files)} | "
                  f"GT={gt_count} Pred={pred_count} Err={error:+d}")

    # Compute metrics
    n = len(abs_errors)
    mae = np.mean(abs_errors) if n > 0 else 0.0
    gt_list = [r["ground_truth"] for r in results]
    mape = (
        np.mean([ae / gt * 100 if gt > 0 else 0.0 for ae, gt in zip(abs_errors, gt_list)])
        if n > 0
        else 0.0
    )
    avg_fps = n / total_inference_time if total_inference_time > 0 else 0.0

    # Count FP/FN frames
    fp_frames = sum(1 for e in errors if e > 0)
    fn_frames = sum(1 for e in errors if e < 0)

    metrics = {
        "dataset": str(dataset_path),
        "mode": mode_label,
        "confidence_threshold": conf,
        "total_frames": n,
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "avg_fps": round(avg_fps, 2),
        "frames_with_overcount": fp_frames,
        "frames_with_undercount": fn_frames,
        "frames_exact": sum(1 for e in errors if e == 0),
        "device": device,
    }

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS — {mode_label}")
    print("=" * 60)
    print(f"  Dataset:              {dataset_path.name}")
    print(f"  Mode:                 {mode_label}")
    print(f"  Confidence Threshold: {conf}")
    print(f"  Total Frames:         {n}")
    print(f"  MAE (per frame):      {metrics['mae']}")
    print(f"  MAPE (per frame):     {metrics['mape']}%")
    print(f"  Avg FPS:              {metrics['avg_fps']}")
    print(f"  Overcount frames:     {fp_frames}")
    print(f"  Undercount frames:    {fn_frames}")
    print(f"  Exact frames:         {metrics['frames_exact']}")
    print("=" * 60)

    # Save results JSON
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({"metrics": metrics, "per_frame": results}, f, indent=2)
        print(f"\n[INFO] Results saved to: {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="People Counting Evaluation (MOT20)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to MOT20 sequence folder")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--save-samples", type=str, default=None, help="Directory to save high-error frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick testing")
    parser.add_argument("--enhance", action="store_true",
                        help="Enable CLAHE + tile-based inference for evaluation")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset path not found: {args.dataset}")
        sys.exit(1)

    evaluate(
        dataset_path=args.dataset,
        conf=args.conf,
        device=args.device,
        output_path=args.output,
        save_samples_dir=args.save_samples,
        max_frames=args.max_frames,
        enhance=args.enhance,
    )


if __name__ == "__main__":
    main()