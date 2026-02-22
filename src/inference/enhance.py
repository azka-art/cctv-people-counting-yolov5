"""
Enhanced preprocessing and inference techniques for improved people detection.

Techniques:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization) - improves detection in
   backlight/low-light conditions common in BRT station CCTV footage.
2. Tile-based inference - splits frame into overlapping patches for better detection
   of small/distant people in crowded scenes.

These mitigations directly address failure cases documented in error_analysis.md:
- Case 1 & 5: Severe occlusion + high density -> tile-based inference
- Case 4: Backlight/poor lighting -> CLAHE preprocessing
"""

import cv2
import numpy as np


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Normalizes local contrast to recover detail in dark/overexposed regions.
    Mitigates error_analysis.md Case 4 (backlight/low-light FN).

    Args:
        image_bgr: Input image in BGR format
        clip_limit: CLAHE clip limit (higher = more contrast)
        grid_size: Tile grid size for adaptive equalization

    Returns:
        Enhanced image in BGR format
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def tile_inference(model, image_rgb: np.ndarray, conf: float = 0.3,
                   tile_size: int = 640, overlap: float = 0.25,
                   iou_threshold: float = 0.3,
                   min_box_area: int = 1500) -> list:
    """
    Tile-based inference for detecting small/distant people.

    Splits the image into overlapping tiles, runs detection on each tile,
    then merges results with aggressive NMS to remove duplicates.

    Mitigates error_analysis.md Case 5 (high density + distant perspective FN).

    Args:
        model: YOLOv5 model
        image_rgb: Input image in RGB format
        conf: Confidence threshold (default 0.3 balances recall vs precision)
        tile_size: Size of each tile in pixels
        overlap: Overlap ratio between adjacent tiles (0.0 - 0.5)
        iou_threshold: IoU threshold for NMS (lower = more aggressive dedup)
        min_box_area: Minimum bounding box area in pixels to keep (filters noise)

    Returns:
        list[dict]: Merged detections with x1, y1, x2, y2, score
    """
    h, w = image_rgb.shape[:2]

    # If image is small enough, no tiling needed
    if h <= tile_size and w <= tile_size:
        dets = _detect_single(model, image_rgb, conf)
        return _filter_small_boxes(dets, min_box_area)

    stride = int(tile_size * (1 - overlap))
    all_detections = []

    # Generate tile coordinates
    for y_start in range(0, h, stride):
        for x_start in range(0, w, stride):
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)

            # Skip very small edge tiles
            if (x_end - x_start) < tile_size * 0.3 or (y_end - y_start) < tile_size * 0.3:
                continue

            tile = image_rgb[y_start:y_end, x_start:x_end]
            tile_dets = _detect_single(model, tile, conf)

            # Offset coordinates back to full image
            for det in tile_dets:
                det["x1"] += x_start
                det["y1"] += y_start
                det["x2"] += x_start
                det["y2"] += y_start
                all_detections.append(det)

    # Also run on full image (catches large subjects better)
    full_dets = _detect_single(model, image_rgb, conf)
    all_detections.extend(full_dets)

    if len(all_detections) == 0:
        return []

    # Filter tiny spurious boxes
    all_detections = _filter_small_boxes(all_detections, min_box_area)

    # Aggressive NMS to remove duplicate detections from overlapping tiles
    merged = _nms_merge(all_detections, iou_threshold)
    return merged


def _detect_single(model, image_rgb: np.ndarray, conf: float) -> list:
    """Run detection on a single image/tile."""
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
                "score": round(float(confidence), 4),
            })
    return detections


def _filter_small_boxes(detections: list, min_area: int = 1500) -> list:
    """Remove detections with bounding box area below threshold."""
    filtered = []
    for det in detections:
        w = det["x2"] - det["x1"]
        h = det["y2"] - det["y1"]
        area = w * h
        if area >= min_area:
            filtered.append(det)
    return filtered


def _nms_merge(detections: list, iou_threshold: float = 0.3) -> list:
    """
    Non-Maximum Suppression to merge overlapping detections from tiles.

    Uses a lower IoU threshold (0.3) for more aggressive deduplication,
    since tile overlaps produce many near-duplicate boxes.

    Args:
        detections: List of detection dicts
        iou_threshold: IoU threshold above which detections are considered duplicates

    Returns:
        Filtered list of detections
    """
    if not detections:
        return []

    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=np.float32)
    scores = np.array([d["score"] for d in detections], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h

        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    result = []
    for idx in keep:
        det = detections[idx].copy()
        det["score"] = round(det["score"], 2)
        result.append(det)

    return result