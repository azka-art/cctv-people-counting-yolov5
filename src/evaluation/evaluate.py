"""
Evaluation script for People Detection & Counting.

Compares YOLOv5s detection counts against MOT20 ground truth annotations
and computes:
- MAE / MAPE (counting accuracy)
- Precision / Recall (detection accuracy, IoU >= 0.5)

Supports --enhance flag for CLAHE + tile-based inference evaluation.

Usage:
    # Standard evaluation
    python -m src.evaluation.evaluate \
        --dataset data/mot20/train/MOT20-01 \
        --conf 0.4 --device cpu \
        --output assets/sample_outputs/eval_results.json \
        --save-samples assets/sample_outputs/

    # Enhanced evaluation
    python -m src.evaluation.evaluate \
        --dataset data/mot20/train/MOT20-01 \
        --conf 0.3 --device cpu --enhance \
        --output assets/sample_outputs/eval_results_enhanced.json \
        --save-samples assets/sample_outputs/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.inference.inference_image import annotate_image, detect_people, load_model


def compute_iou(box_a, box_b):
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    """
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])

    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def match_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match predicted boxes to ground truth boxes using greedy IoU matching.

    Args:
        pred_boxes: list of [x1, y1, x2, y2]
        gt_boxes: list of [x1, y1, x2, y2]
        iou_threshold: minimum IoU for a match (default 0.5)

    Returns:
        tp: number of true positives
        fp: number of false positives
        fn: number of false negatives
    """
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pb, gb)

    # Greedy matching: highest IoU first
    matched_pred = set()
    matched_gt = set()

    # Flatten and sort by IoU descending
    pairs = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    for _, pred_idx, gt_idx in pairs:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


def load_mot_ground_truth_counts(gt_path: str) -> dict:
    """
    Load MOT format ground truth and count unique persons per frame.

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

            if conf == 0:
                continue
            if cls not in (1, 2):
                continue

            gt_counts[frame_id] = gt_counts.get(frame_id, 0) + 1

    return gt_counts


def load_mot_ground_truth_boxes(gt_path: str) -> dict:
    """
    Load MOT format ground truth bounding boxes per frame.

    MOT format: frame, id, x, y, w, h, conf, cls, vis
    Coordinates: x,y = top-left corner; w,h = width,height

    Returns:
        dict mapping frame_number -> list of [x1, y1, x2, y2]
    """
    gt_boxes = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            frame_id = int(parts[0])
            conf = int(parts[6])
            cls = int(parts[7])

            if conf == 0:
                continue
            if cls not in (1, 2):
                continue

            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            box = [x, y, x + w, y + h]

            if frame_id not in gt_boxes:
                gt_boxes[frame_id] = []
            gt_boxes[frame_id].append(box)

    return gt_boxes


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

    Computes:
    - MAE / MAPE (counting accuracy)
    - Precision / Recall (detection accuracy with IoU >= 0.5 matching)
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

    # Load ground truth (counts + boxes)
    gt_counts = load_mot_ground_truth_counts(str(gt_path))
    gt_boxes_all = load_mot_ground_truth_boxes(str(gt_path))
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

    if enhance:
        from src.inference.enhance import apply_clahe, tile_inference

    # Evaluate frame by frame
    results = []
    errors = []
    abs_errors = []
    total_inference_time = 0.0

    # Detection metrics accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path in image_files:
        frame_num = int(img_path.stem)
        gt_count = gt_counts.get(frame_num, 0)
        gt_boxes = gt_boxes_all.get(frame_num, [])

        # Load frame
        frame_bgr = cv2.imread(str(img_path))

        start_time = time.time()

        if enhance:
            enhanced_bgr = apply_clahe(frame_bgr)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            detections = tile_inference(model, enhanced_rgb, conf=conf)
            pred_count = len(detections)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections, pred_count = detect_people(model, frame_rgb, conf)

        inference_time = time.time() - start_time
        total_inference_time += inference_time

        # Counting error
        error = pred_count - gt_count
        abs_error = abs(error)

        # Detection matching (precision/recall)
        pred_boxes = [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections]
        tp, fp, fn = match_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        results.append({
            "frame": frame_num,
            "ground_truth": gt_count,
            "predicted": pred_count,
            "error": error,
            "abs_error": abs_error,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
        errors.append(error)
        abs_errors.append(abs_error)

        # Save frames with high error for error analysis
        if save_samples_dir and abs_error > 2:
            save_dir = Path(save_samples_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            annotated = annotate_image(frame_bgr, detections, pred_count)

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
                  f"GT={gt_count} Pred={pred_count} Err={error:+d} | "
                  f"TP={tp} FP={fp} FN={fn}")

    # Compute counting metrics
    n = len(abs_errors)
    mae = np.mean(abs_errors) if n > 0 else 0.0
    gt_list = [r["ground_truth"] for r in results]
    mape = (
        np.mean([ae / gt * 100 if gt > 0 else 0.0 for ae, gt in zip(abs_errors, gt_list)])
        if n > 0
        else 0.0
    )
    avg_fps = n / total_inference_time if total_inference_time > 0 else 0.0

    # Compute detection metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

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
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "avg_fps": round(avg_fps, 2),
        "frames_with_overcount": fp_frames,
        "frames_with_undercount": fn_frames,
        "frames_exact": sum(1 for e in errors if e == 0),
        "device": device,
    }

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS -- {mode_label}")
    print("=" * 60)
    print(f"  Dataset:              {dataset_path.name}")
    print(f"  Mode:                 {mode_label}")
    print(f"  Confidence Threshold: {conf}")
    print(f"  Total Frames:         {n}")
    print(f"")
    print(f"  --- Counting Metrics ---")
    print(f"  MAE (per frame):      {metrics['mae']}")
    print(f"  MAPE (per frame):     {metrics['mape']}%")
    print(f"  Overcount frames:     {fp_frames}")
    print(f"  Undercount frames:    {fn_frames}")
    print(f"  Exact frames:         {metrics['frames_exact']}")
    print(f"")
    print(f"  --- Detection Metrics (IoU >= 0.5) ---")
    print(f"  Precision:            {metrics['precision']:.4f}")
    print(f"  Recall:               {metrics['recall']:.4f}")
    print(f"  F1 Score:             {metrics['f1_score']:.4f}")
    print(f"  Total TP:             {total_tp}")
    print(f"  Total FP:             {total_fp}")
    print(f"  Total FN:             {total_fn}")
    print(f"")
    print(f"  --- Performance ---")
    print(f"  Avg FPS:              {metrics['avg_fps']}")
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