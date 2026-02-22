# Model Card -- People Detection & Counting

> **Version:** 1.2 | **Date:** 2025 | **Framework:** PyTorch (YOLOv5s via `torch.hub`)

---

## 1. Model Overview

| Attribute | Detail |
|---|---|
| **Task** | Object Detection + Per-frame People Counting + Tracking |
| **Architecture** | YOLOv5s (Small variant -- efficient, low latency) |
| **Pretrained On** | COCO 2017 (80 classes, `person` = Class 0) |
| **Model Source** | `torch.hub.load('ultralytics/yolov5', 'yolov5s')` |
| **Output** | Bounding boxes, confidence scores, People Count overlay, track IDs |
| **Target Class** | `person` only (other classes filtered in post-processing) |
| **Default Confidence** | `0.4` (standard) / `0.3` (enhanced) |

---

## 2. Intended Use

### 2.1 Suitable Scenarios

- **BRT station density monitoring:** Count people in waiting areas or corridors per frame.
- **Queue throughput analysis:** Measure passenger flow on static elevated-angle cameras.
- **End-to-end pipeline demo:** CV-to-REST-API integration in one repository.
- **Fleet capacity evaluation:** Early load monitoring signal for high-volume stations.

### 2.2 Out-of-Scope

- Biometric identification or face recognition.
- Identity tracking for law enforcement or surveillance.
- Official statistical reporting without spatial calibration and manual validation.
- Critical production deployment without bias audit and domain-specific accuracy validation.

---

## 3. Pipeline Inference

### 3.1 Standard Mode

| Step | Tools | Description |
|---|---|---|
| Load image | `PIL.Image.open()` -> RGB conversion | Used in `inference_image.py` |
| Load video frame | `cv2.VideoCapture` -> BGR to RGB | Used in `inference_video.py` |
| Auto resize | YOLOv5 internal (letterbox) | Model handles resize to 640x640 |
| Filter class | Post-processing | Only class 0 (person) |

### 3.2 Enhanced Mode

| Step | Tools | Description |
|---|---|---|
| CLAHE preprocessing | `cv2.createCLAHE` on L-channel (LAB) | Local contrast normalization |
| Tile-based inference | 640px tiles, 25% overlap | Detect small/distant people |
| Aggressive NMS | IoU threshold 0.3 | Remove duplicates from tile overlap |
| Min area filter | 1500px2 minimum | Remove noise/spurious detections |

### 3.3 Tracking Mode

| Step | Tools | Description |
|---|---|---|
| IoU matching | `SimpleTracker` (SORT-lite) | Match detections to existing tracks |
| ID assignment | Greedy highest-IoU-first | Persistent unique IDs per person |
| Track management | max_disappeared=fps | Remove lost tracks after ~1 sec |

### 3.4 Output Format -- API Contract

Endpoint `POST /detect/image` returns:

```json
{
  "count": 28,
  "detections": [
    { "x1": 15,  "y1": 30,  "x2": 110, "y2": 240, "score": 0.88 },
    { "x1": 200, "y1": 45,  "x2": 310, "y2": 250, "score": 0.76 }
  ]
}
```

---

## 4. Performance Metrics

### 4.1 Counting Accuracy (MAE / MAPE)

| Metric | Standard (conf=0.4) | Enhanced (conf=0.3) | Change |
|---|---|---|---|
| **MAE** | 32.38 | **10.71** | **-67%** |
| **MAPE** | 69.78% | **22.81%** | **-67%** |
| Overcount frames | 0 | 0 | -- |
| Undercount frames | 429 (100%) | 426 | -- |
| Exact match frames | 0 | 3 | -- |

### 4.2 Detection Quality (IoU >= 0.5 matching)

| Metric | Standard (conf=0.4) | Enhanced (conf=0.3) | Change |
|---|---|---|---|
| **Precision** | **0.9831** | 0.7141 | Trade-off |
| **Recall** | 0.2958 | **0.5490** | **+86%** |
| **F1 Score** | 0.4547 | **0.6208** | **+37%** |
| Total TP | 5,877 | 10,909 | +86% |
| Total FP | 101 | 4,367 | Trade-off |
| Total FN | 13,993 | 8,961 | -36% |

**Analysis:** Standard mode achieves near-perfect precision (98.3%) but misses ~70% of people (recall 29.6%). Enhanced mode dramatically improves recall to 54.9% by using CLAHE + tile inference, at the cost of lower precision (71.4%). The F1 score improves by 37%, confirming that the recall gain outweighs the precision trade-off for crowded transit monitoring use cases.

### 4.3 Performance

| Metric | Standard | Enhanced |
|---|---|---|
| **Avg FPS** | 3.59 | 0.31 |
| **Device** | CPU | CPU |
| **Dataset** | MOT20-01 (429 frames) | MOT20-01 (429 frames) |
| **Hardware** | Intel Core CPU | Intel Core CPU |

---

## 5. Limitations

| # | Type | Condition | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **False Negative** | Severe occlusion (>70% body occluded) | Undercount at crush load | Tile inference |
| 2 | **False Negative** | Motion blur (speed >= 2 m/s) | Missed during door opening | -- |
| 3 | **False Positive** | Poster/ads with human figures | Overcount near displays | Min area filter |
| 4 | **False Negative** | Poor lighting (night / backlight) | Failed detection at dark stops | CLAHE |
| 5 | **False Negative** | High density + distant perspective | 50% undercount at elevated cameras | Tile inference |

---

## 6. Mitigations & Future Work

| Priority | Step | Impact | Status |
|---|---|---|---|
| **High** | CLAHE preprocessing | Reduce FN in backlight/dark | Done |
| **High** | Tile-based inference | Recall 0.296 -> 0.549 | Done |
| **High** | Aggressive NMS + min area filter | Reduce FP from tile overlap | Done |
| **High** | IoU-based tracking (SORT-lite) | Persistent IDs, unique counting | Done |
| **High** | Precision/Recall evaluation | Detection quality beyond counting | Done |
| **Medium** | Fine-tuning on CCTV transport data | Better accuracy for halte conditions | Future |
| **Medium** | Virtual tripwire / line crossing | Accurate entry/exit counting | Future |
| **Low** | Model upgrade to YOLOv8m/YOLOv9 | Higher mAP on dense crowds | Future |

---

## 7. Ethical Considerations

This repository is built for **aggregate capacity monitoring**, not individual surveillance. Deployment in public environments like TransJakarta must consider:

- Transparency to passengers that a counting system is active.
- No storage of footage or bounding boxes that could identify individuals.
- Compliance with applicable personal data protection regulations.