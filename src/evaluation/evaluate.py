"""
Evaluation script for People Detection & Counting.

Compares YOLOv5s detection counts against MOT20 ground truth annotations
and computes:
  - MAE / MAPE            (counting accuracy)
  - Precision / Recall / F1  (detection accuracy, IoU >= 0.5)
  - IDSW / MOTA           (tracking quality, when --track is enabled)

Supports:
  --enhance   CLAHE + tile-based inference
  --track     SORTTracker evaluation with ID switch and MOTA metrics

Usage:
    # Standard evaluation
    python -m src.evaluation.evaluate \\
        --dataset data/mot20/train/MOT20-01 \\
        --conf 0.4 --device cpu \\
        --output assets/sample_outputs/eval_results.json \\
        --save-samples assets/sample_outputs/

    # Enhanced evaluation
    python -m src.evaluation.evaluate \\
        --dataset data/mot20/train/MOT20-01 \\
        --conf 0.3 --device cpu --enhance \\
        --output assets/sample_outputs/eval_results_enhanced.json \\
        --save-samples assets/sample_outputs/

    # With tracking metrics (IDSW, MOTA)
    python -m src.evaluation.evaluate \\
        --dataset data/mot20/train/MOT20-01 \\
        --conf 0.3 --device cpu --enhance --track \\
        --output assets/sample_outputs/eval_results_tracked.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.inference.inference_image import annotate_image, detect_people, load_model


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def compute_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])

    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def match_detections(
    pred_boxes: list,
    gt_boxes: list,
    iou_threshold: float = 0.5,
) -> tuple:
    """
    Match predicted boxes to ground truth boxes via greedy IoU matching.

    Args:
        pred_boxes:     list of [x1, y1, x2, y2]
        gt_boxes:       list of [x1, y1, x2, y2]
        iou_threshold:  minimum IoU for a valid match (default: 0.5)

    Returns:
        tp, fp, fn  (int, int, int)
    """
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pb, gb)

    matched_pred = set()
    matched_gt   = set()
    pairs = [
        (iou_matrix[i, j], i, j)
        for i in range(len(pred_boxes))
        for j in range(len(gt_boxes))
        if iou_matrix[i, j] >= iou_threshold
    ]
    pairs.sort(reverse=True)

    for _, pred_idx, gt_idx in pairs:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes)   - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Ground truth loaders
# ---------------------------------------------------------------------------

def load_mot_ground_truth_counts(gt_path: str) -> dict:
    """
    Load MOT format ground truth and count unique persons per frame.

    Returns:
        dict: frame_number -> person_count
    """
    gt_counts = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            conf     = int(parts[6])
            cls      = int(parts[7])
            if conf == 0 or cls not in (1, 2):
                continue
            gt_counts[frame_id] = gt_counts.get(frame_id, 0) + 1
    return gt_counts


def load_mot_ground_truth_boxes(gt_path: str) -> dict:
    """
    Load MOT format ground truth bounding boxes per frame.

    MOT format: frame, id, x, y, w, h, conf, cls, vis
    Coordinates: x,y = top-left corner; w,h = width, height

    Returns:
        dict: frame_number -> list of [x1, y1, x2, y2]
    """
    gt_boxes = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            conf     = int(parts[6])
            cls      = int(parts[7])
            if conf == 0 or cls not in (1, 2):
                continue

            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if frame_id not in gt_boxes:
                gt_boxes[frame_id] = []
            gt_boxes[frame_id].append([x, y, x + w, y + h])
    return gt_boxes


def load_mot_ground_truth_ids(gt_path: str) -> dict:
    """
    Load MOT ground truth per-frame with object IDs for tracking evaluation.

    Returns:
        dict: frame_number -> list of (object_id, [x1, y1, x2, y2])
    """
    gt_ids = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame_id  = int(parts[0])
            object_id = int(parts[1])
            conf      = int(parts[6])
            cls       = int(parts[7])
            if conf == 0 or cls not in (1, 2):
                continue

            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if frame_id not in gt_ids:
                gt_ids[frame_id] = []
            gt_ids[frame_id].append((object_id, [x, y, x + w, y + h]))
    return gt_ids


# ---------------------------------------------------------------------------
# Tracking metrics computation
# ---------------------------------------------------------------------------

def compute_tracking_metrics(
    all_frame_results: list,
    gt_ids_all: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute SORT-style tracking metrics over the full evaluation sequence.

    Metrics computed:
        IDSW  — ID switches: number of times a GT object changes its assigned
                predicted track ID across consecutive frames
        MOTA  — Multiple Object Tracking Accuracy:
                MOTA = 1 - (FP + FN + IDSW) / total_GT_objects
                Range: (-inf, 1.0]; higher is better; > 0 is considered usable

    Args:
        all_frame_results:  list of per-frame dicts from evaluate() loop,
                            each containing 'frame', 'tracks' (list of track dicts)
        gt_ids_all:         output of load_mot_ground_truth_ids()
        iou_threshold:      IoU threshold for GT↔track matching (default: 0.5)

    Returns:
        dict with keys: id_switches, mota, total_gt_objects,
                        idsw_rate_pct, mota_interpretation
    """
    # gt_id -> last assigned track_id
    gt_to_track: dict = {}
    id_switches  = 0
    total_gt     = 0
    total_fp_tracking = 0
    total_fn_tracking = 0

    for frame_result in all_frame_results:
        frame_num = frame_result["frame"]
        tracks    = frame_result.get("tracks", [])

        gt_entries = gt_ids_all.get(frame_num, [])
        total_gt  += len(gt_entries)

        gt_boxes_frame = [entry[1] for entry in gt_entries]
        gt_obj_ids     = [entry[0] for entry in gt_entries]

        track_boxes = [[t["x1"], t["y1"], t["x2"], t["y2"]] for t in tracks]
        track_ids   = [t["track_id"] for t in tracks]

        # Match GT objects to predicted tracks via IoU
        if len(gt_boxes_frame) == 0 or len(track_boxes) == 0:
            total_fn_tracking += len(gt_boxes_frame)
            total_fp_tracking += len(track_boxes)
            continue

        iou_matrix = np.zeros((len(gt_boxes_frame), len(track_boxes)))
        for i, gb in enumerate(gt_boxes_frame):
            for j, tb in enumerate(track_boxes):
                iou_matrix[i, j] = compute_iou(gb, tb)

        matched_gt    = set()
        matched_track = set()
        pairs = [
            (iou_matrix[i, j], i, j)
            for i in range(len(gt_boxes_frame))
            for j in range(len(track_boxes))
            if iou_matrix[i, j] >= iou_threshold
        ]
        pairs.sort(reverse=True)

        for _, gt_idx, track_idx in pairs:
            if gt_idx in matched_gt or track_idx in matched_track:
                continue
            matched_gt.add(gt_idx)
            matched_track.add(track_idx)

            gt_obj_id  = gt_obj_ids[gt_idx]
            pred_tid   = track_ids[track_idx]

            # ID switch: same GT object, different predicted track ID
            if gt_obj_id in gt_to_track:
                if gt_to_track[gt_obj_id] != pred_tid:
                    id_switches += 1
            gt_to_track[gt_obj_id] = pred_tid

        total_fn_tracking += len(gt_boxes_frame) - len(matched_gt)
        total_fp_tracking += len(track_boxes)    - len(matched_track)

    # MOTA = 1 - (FP + FN + IDSW) / total_GT
    mota = (
        1.0 - (total_fp_tracking + total_fn_tracking + id_switches) / total_gt
        if total_gt > 0 else 0.0
    )
    idsw_rate = id_switches / total_gt * 100 if total_gt > 0 else 0.0

    if mota >= 0.5:
        mota_interp = "Good (>= 0.5)"
    elif mota >= 0.2:
        mota_interp = "Fair (0.2–0.5)"
    elif mota >= 0.0:
        mota_interp = "Poor (0.0–0.2)"
    else:
        mota_interp = "Very poor (< 0)"

    return {
        "id_switches":        id_switches,
        "idsw_rate_pct":      round(idsw_rate, 2),
        "mota":               round(mota, 4),
        "mota_interpretation": mota_interp,
        "total_gt_objects":   total_gt,
        "tracking_fp":        total_fp_tracking,
        "tracking_fn":        total_fn_tracking,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    dataset_path: str,
    conf: float = 0.4,
    device: str = "cpu",
    output_path: str = None,
    save_samples_dir: str = None,
    max_frames: int = None,
    enhance: bool = False,
    track: bool = False,
) -> dict:
    """
    Run full evaluation on a MOT20 sequence.

    Computes counting metrics (MAE/MAPE), detection metrics (P/R/F1),
    and optionally tracking metrics (IDSW/MOTA) when --track is enabled.

    Args:
        dataset_path:     Path to MOT20 sequence folder (contains img1/ and gt/)
        conf:             YOLOv5 confidence threshold
        device:           "cpu" or "cuda"
        output_path:      Path to save JSON results (optional)
        save_samples_dir: Directory to save annotated high-error frames (optional)
        max_frames:       Limit frames evaluated (for quick testing)
        enhance:          Enable CLAHE + tile-based inference
        track:            Enable SORTTracker and compute IDSW/MOTA metrics

    Returns:
        dict of all computed metrics
    """
    dataset_path = Path(dataset_path)
    img_dir  = dataset_path / "img1"
    gt_path  = dataset_path / "gt" / "gt.txt"

    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        sys.exit(1)

    # Load ground truth
    gt_counts    = load_mot_ground_truth_counts(str(gt_path))
    gt_boxes_all = load_mot_ground_truth_boxes(str(gt_path))
    gt_ids_all   = load_mot_ground_truth_ids(str(gt_path)) if track else {}
    print(f"[INFO] Ground truth loaded: {len(gt_counts)} annotated frames")

    # Image files
    image_files = sorted(img_dir.glob("*.jpg"))
    if max_frames:
        image_files = image_files[:max_frames]

    mode_label = "Enhanced (CLAHE + tile)" if enhance else "Standard"
    track_label = " + Tracking (SORT)" if track else ""
    print(f"[INFO] Evaluating {len(image_files)} frames")
    print(f"[INFO] Mode      : {mode_label}{track_label}")
    print(f"[INFO] Confidence: {conf}  |  Device: {device}")
    print()

    # Load model + optional modules
    model = load_model(device)
    if enhance:
        from src.inference.enhance import apply_clahe, tile_inference

    tracker = None
    if track:
        from src.inference.tracker import SORTTracker
        tracker = SORTTracker(max_age=30, min_hits=1, iou_threshold=0.3)

    # ----------------------------------------------------------------
    # Frame loop
    # ----------------------------------------------------------------
    results              = []
    errors               = []
    abs_errors           = []
    total_inference_time = 0.0
    total_tp = total_fp = total_fn = 0

    for img_path in image_files:
        frame_num = int(img_path.stem)
        gt_count  = gt_counts.get(frame_num, 0)
        gt_boxes  = gt_boxes_all.get(frame_num, [])

        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        t0 = time.perf_counter()

        if enhance:
            enhanced_bgr = apply_clahe(frame_bgr)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            detections   = tile_inference(model, enhanced_rgb, conf=conf)
            pred_count   = len(detections)
        else:
            frame_rgb          = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections, pred_count = detect_people(model, frame_rgb, conf)

        inference_time        = time.perf_counter() - t0
        total_inference_time += inference_time

        # Tracking update
        tracks = []
        if track and tracker is not None:
            tracks = tracker.update(detections)

        # Counting error
        error     = pred_count - gt_count
        abs_error = abs(error)
        errors.append(error)
        abs_errors.append(abs_error)

        # Detection metrics (IoU >= 0.5)
        pred_boxes = [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections]
        tp, fp, fn = match_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        frame_result = {
            "frame":        frame_num,
            "ground_truth": gt_count,
            "predicted":    pred_count,
            "error":        error,
            "abs_error":    abs_error,
            "tp":           tp,
            "fp":           fp,
            "fn":           fn,
        }
        if track:
            frame_result["tracks"] = tracks

        results.append(frame_result)

        # Save high-error frames
        if save_samples_dir and abs_error > 2:
            save_dir = Path(save_samples_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            annotated = annotate_image(frame_bgr, detections, pred_count)
            gt_text   = f"GT: {gt_count} | Pred: {pred_count} | Err: {error:+d}"
            cv2.putText(
                annotated, gt_text, (15, frame_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            suffix    = "_enhanced" if enhance else "_standard"
            save_path = save_dir / f"error_frame_{frame_num:06d}{suffix}.jpg"
            cv2.imwrite(str(save_path), annotated)

        # Progress log every 10 frames
        frame_idx = len(results)
        if frame_idx % 10 == 0 or frame_idx == len(image_files):
            track_str = ""
            if track and tracker:
                m = tracker.get_metrics()
                track_str = (
                    f" | Unique: {m['total_unique']}"
                    f"  IDSW: {m['id_switches']}"
                )
            print(
                f"[INFO] Frame {frame_idx:>4}/{len(image_files)}"
                f"  GT={gt_count:>3}  Pred={pred_count:>3}  Err={error:>+4d}"
                f"  TP={tp}  FP={fp}  FN={fn}"
                f"{track_str}"
            )

    # ----------------------------------------------------------------
    # Aggregate metrics
    # ----------------------------------------------------------------
    n    = len(abs_errors)
    mae  = float(np.mean(abs_errors)) if n > 0 else 0.0
    gt_list = [r["ground_truth"] for r in results]
    mape = float(np.mean([
        abs_errors[i] / gt_list[i] * 100 if gt_list[i] > 0 else 0.0
        for i in range(n)
    ])) if n > 0 else 0.0
    avg_fps = n / total_inference_time if total_inference_time > 0 else 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    fp_frames     = sum(1 for e in errors if e > 0)
    fn_frames     = sum(1 for e in errors if e < 0)
    exact_frames  = sum(1 for e in errors if e == 0)

    metrics = {
        "dataset":              str(dataset_path),
        "mode":                 mode_label + track_label,
        "confidence_threshold": conf,
        "total_frames":         n,
        # Counting
        "mae":                  round(mae, 2),
        "mape":                 round(mape, 2),
        "frames_with_overcount":  fp_frames,
        "frames_with_undercount": fn_frames,
        "frames_exact":           exact_frames,
        # Detection
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "f1_score":             round(f1, 4),
        "total_tp":             total_tp,
        "total_fp":             total_fp,
        "total_fn":             total_fn,
        # Performance
        "avg_fps":              round(avg_fps, 2),
        "device":               device,
    }

    # Tracking metrics (only when --track)
    tracking_metrics = {}
    if track and tracker:
        tracking_metrics = compute_tracking_metrics(
            results, gt_ids_all, iou_threshold=0.5
        )
        metrics.update(tracking_metrics)

    # ----------------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------------
    _print_summary(metrics, tracking_metrics, mode_label, track_label)

    # Save results JSON
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serializable track dicts from per-frame before saving
        per_frame_clean = [
            {k: v for k, v in r.items() if k != "tracks"}
            for r in results
        ]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "per_frame": per_frame_clean}, f, indent=2)
        print(f"\n[INFO] Results saved to: {output_path}")

    return metrics


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def _print_summary(metrics: dict, tracking_metrics: dict, mode: str, track_label: str):
    w = 60
    print()
    print("=" * w)
    print(f"  EVALUATION RESULTS — {mode}{track_label}")
    print("=" * w)
    print(f"  Dataset   : {Path(metrics['dataset']).name}")
    print(f"  Conf      : {metrics['confidence_threshold']}")
    print(f"  Frames    : {metrics['total_frames']}")
    print()
    print("  ── Counting Metrics ────────────────────────────────")
    print(f"  MAE  (per frame)    : {metrics['mae']}")
    print(f"  MAPE (per frame)    : {metrics['mape']}%")
    print(f"  Overcount  frames   : {metrics['frames_with_overcount']}")
    print(f"  Undercount frames   : {metrics['frames_with_undercount']}")
    print(f"  Exact      frames   : {metrics['frames_exact']}")
    print()
    print("  ── Detection Metrics (IoU >= 0.5) ──────────────────")
    print(f"  Precision           : {metrics['precision']:.4f}")
    print(f"  Recall              : {metrics['recall']:.4f}")
    print(f"  F1 Score            : {metrics['f1_score']:.4f}")
    print(f"  TP / FP / FN        : {metrics['total_tp']} / "
          f"{metrics['total_fp']} / {metrics['total_fn']}")
    print()
    print("  ── Performance ──────────────────────────────────────")
    print(f"  Avg FPS             : {metrics['avg_fps']}")
    print(f"  Device              : {metrics['device']}")

    if tracking_metrics:
        print()
        print("  ── Tracking Metrics (SORT) ──────────────────────────")
        print(f"  ID Switches (IDSW)  : {tracking_metrics['id_switches']}")
        print(f"  IDSW Rate           : {tracking_metrics['idsw_rate_pct']}%  "
              f"(IDSW / total GT objects)")
        print(f"  MOTA                : {tracking_metrics['mota']:.4f}  "
              f"→ {tracking_metrics['mota_interpretation']}")
        print(f"  Total GT objects    : {tracking_metrics['total_gt_objects']}")
        print()
        print("  MOTA = 1 − (FP + FN + IDSW) / Σ GT objects")
        print("  Range: (−∞, 1.0] — higher is better; > 0 is usable")

    print("=" * w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="People Detection & Counting — Evaluation (MOT20)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to MOT20 sequence folder\n"
             "(must contain img1/ and gt/gt.txt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results\n"
             "(e.g. assets/sample_outputs/eval_results.json)",
    )
    parser.add_argument(
        "--save-samples", type=str, default=None,
        help="Directory to save annotated high-error frames\n"
             "(frames where |error| > 2)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit number of frames for quick testing",
    )
    parser.add_argument(
        "--enhance", action="store_true",
        help="Enable CLAHE + tile-based inference\n"
             "(better recall in crowded/backlit scenes)",
    )
    parser.add_argument(
        "--track", action="store_true",
        help="Enable SORTTracker and compute tracking metrics\n"
             "(IDSW, MOTA) — requires filterpy",
    )

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
        track=args.track,
    )


if __name__ == "__main__":
    main()