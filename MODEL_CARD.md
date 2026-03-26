# Model Card — People Detection, Counting & Tracking

> **Version:** 1.3 | **Date:** March 2026 | **Framework:** PyTorch (YOLOv5s via `torch.hub`)

---

## 1. Model Overview

| Attribute | Detail |
|---|---|
| **Task** | Object Detection · Per-frame People Counting · Multi-object Tracking |
| **Architecture** | YOLOv5s (Small variant — efficient, low latency) |
| **Pretrained On** | COCO 2017 (80 classes, `person` → Class 0) |
| **Model Source** | `torch.hub.load('ultralytics/yolov5', 'yolov5s')` |
| **Export Format** | PyTorch `.pt` + ONNX (opset 11, verified with ONNXRuntime) |
| **ONNX File Size** | 0.31 MB |
| **Output** | Bounding boxes · confidence scores · People Count overlay · persistent track IDs |
| **Target Class** | `person` only (all other COCO classes filtered in post-processing) |
| **Default Confidence** | `0.4` (standard) · `0.3` (enhanced / tracking) |

---

## 2. Intended Use

### 2.1 Suitable Scenarios

- **BRT station density monitoring** — count people in waiting areas or corridors per frame.
- **Queue throughput analysis** — measure passenger flow on static elevated-angle cameras.
- **Tracking stability evaluation** — measure IDSW and MOTA for tracking pipeline QA.
- **End-to-end pipeline demo** — CV inference → REST API integration in one repository.
- **Fleet capacity signal** — early load monitoring at high-volume stations.

### 2.2 Out-of-Scope

- ❌ Biometric identification or face recognition.
- ❌ Identity tracking for law enforcement or surveillance.
- ❌ Official statistical reporting without spatial calibration and manual validation.
- ❌ Critical production deployment without domain-specific fine-tuning and bias audit.

---

## 3. Inference Pipeline

### 3.1 Standard Mode

| Step | Tool | Description |
|---|---|---|
| Load image | `PIL.Image.open()` → RGB | Used in `inference_image.py` |
| Load video frame | `cv2.VideoCapture` → BGR to RGB | Used in `inference_video.py` |
| Auto resize | YOLOv5 internal letterbox | Resize to 640×640 |
| Class filter | Post-processing | Person class (0) only |

### 3.2 Enhanced Mode (`--enhance`)

| Step | Tool | Description |
|---|---|---|
| CLAHE preprocessing | `cv2.createCLAHE` on L-channel (LAB) | Local contrast normalization for backlit/dark frames |
| Tile-based inference | 640px tiles, 25% overlap | Detect small/distant people missed by full-image pass |
| Aggressive NMS | IoU threshold 0.3 | Remove duplicates from tile overlap |
| Min area filter | 1 500 px² minimum box | Remove noise and spurious micro-detections |

### 3.3 Tracking Mode (`--track`)

| Step | Tool | Description |
|---|---|---|
| Kalman prediction | `filterpy.KalmanFilter` (8D state) | Predict next position to bridge detection gaps |
| IoU association | Greedy, highest-IoU-first | Match detections to predicted track positions |
| Track management | `max_age=30` frames (~1 s at 30 fps) | Delete lost tracks after ~1 second without match |
| ID assignment | Monotonically increasing integers | Persistent unique IDs per person |
| ID switch detection | GT object ↔ track ID comparison | Count IDSW for tracking quality evaluation |

### 3.4 ONNX Export

Exported using `torch.onnx.export` with `dynamic_axes` for variable batch size and resolution:

```bash
python -m src.inference.export_onnx --output assets/yolov5s.onnx --verify
```

| Attribute | Value |
|---|---|
| Opset | 11 |
| Input name | `images` |
| Output name | `output` |
| Output shape | `(1, 25200, 85)` |
| File size | 0.31 MB |
| Verified with | ONNXRuntime 1.24.4 |

### 3.5 API Contract

`POST /detect/image` → `multipart/form-data`

```json
{
  "count": 28,
  "detections": [
    { "x1": 15,  "y1": 30,  "x2": 110, "y2": 240, "score": 0.88 },
    { "x1": 200, "y1": 45,  "x2": 310, "y2": 250, "score": 0.76 }
  ]
}
```

> `count` = active detections in the current frame (not tracked across frames).
> Coordinates are in pixels relative to the input image resolution.

---

## 4. Performance Metrics

> **Dataset:** MOT20-01 (MOTChallenge) · **Frames:** 429 · **Hardware:** Intel Core CPU (no GPU)
> **Reproducibility commit:** see `assets/sample_outputs/eval_results*.json`

### 4.1 Counting Accuracy

| Metric | Standard (conf=0.4) | Enhanced (conf=0.3) | Δ |
|---|---|---|---|
| **MAE (per frame)** | 32.38 | **10.71** | ↓ 67% |
| **MAPE (per frame)** | 69.78% | **22.81%** | ↓ 67% |
| Overcount frames | 0 | 0 | — |
| Undercount frames | 429 / 429 | 426 / 429 | — |
| Exact match frames | 0 | 3 | — |

### 4.2 Detection Quality (IoU ≥ 0.5)

| Metric | Standard (conf=0.4) | Enhanced (conf=0.3) | Δ |
|---|---|---|---|
| **Precision** | **0.9831** | 0.7141 | trade-off |
| **Recall** | 0.2958 | **0.5490** | ↑ 86% |
| **F1 Score** | 0.4547 | **0.6208** | ↑ 37% |
| Total TP | 5,877 | 10,909 | ↑ 86% |
| Total FP | 101 | 4,367 | trade-off |
| Total FN | 13,993 | 8,961 | ↓ 36% |

**Design rationale:** Standard mode achieves near-perfect precision (98.3%) but misses ~70% of people in dense crowds (recall 29.6%). Enhanced mode improves recall to 54.9% through CLAHE + tile inference, at the cost of lower precision (71.4%). For crowded BRT station monitoring where undercounting is the dominant failure mode, the 37% F1 improvement and 67% MAE reduction confirm that the recall gain outweighs the precision trade-off.

### 4.3 Tracking Metrics (SORT + Kalman, Enhanced Mode)

> Run with `--enhance --track --conf 0.3` on MOT20-01 (429 frames, 19,870 total GT objects)

| Metric | Value | Interpretation |
|---|---|---|
| **ID Switches (IDSW)** | 318 | Times a GT person changed assigned track ID |
| **IDSW Rate** | **1.6%** | IDSW / total GT objects — low = stable tracker |
| **MOTA** | −0.0323 | See note below |
| Total GT objects | 19,870 | Sum of annotated persons across all frames |

> **IDSW Rate of 1.6% indicates the Kalman-based tracker is highly stable** — the majority of tracks are maintained consistently across the sequence. The low IDSW rate is the most operationally relevant tracking metric for BRT station monitoring, as it reflects how reliably the system maintains identity across frames.

> **Note on MOTA = −0.0323:** MOTA (Multiple Object Tracking Accuracy) is negative because MOT20-01 is an extremely dense scene (average ~46 persons/frame) and the model is a pretrained COCO baseline without domain fine-tuning. Enhanced tile inference improves recall dramatically (+86%) but also increases false positives from tile overlap, which weighs on the MOTA formula: `MOTA = 1 − (FP + FN + IDSW) / Σ GT`. This is expected and documented behavior for a non-fine-tuned model on a high-density benchmark. Fine-tuning on domain-specific CCTV data is the primary recommended next step (see Section 6).

### 4.4 Processing Speed

| Mode | Avg FPS | Notes |
|---|---|---|
| Standard | 3.59 | CPU, no CLAHE/tile |
| Enhanced | 0.31 | CPU, CLAHE + tile (4 tiles/frame) |
| Enhanced + Tracking | ~0.31 | Kalman overhead negligible vs. tile inference |

> GPU acceleration (ONNX + TensorRT) is the recommended path for real-time enhanced mode deployment.

---

## 5. Failure Modes

| # | Type | Condition | Magnitude | Mitigation |
|---|---|---|---|---|
| 1 | **False Negative** | Severe occlusion (>70% body occluded) | −4 per frame | Tile inference |
| 2 | **False Negative** | Motion blur (speed ≥ 2 m/s) | −2 per frame | Temporal averaging |
| 3 | **False Positive** | Poster/ad with human figures on wall | +2 per frame | Min area filter · ROI masking |
| 4 | **False Negative** | Poor lighting / backlight (dawn/dusk) | −5 per frame | CLAHE |
| 5 | **False Negative** | High density + elevated bird-eye distance | −11 per frame | Tile inference |

See [`src/evaluation/error_analysis.md`](src/evaluation/error_analysis.md) for full analysis with visual evidence.

---

## 6. Mitigations & Future Work

| Priority | Step | Measured Impact | Status |
|---|---|---|---|
| ✅ Done | CLAHE preprocessing | Recovers silhouette detail in backlit frames | Done |
| ✅ Done | Tile-based inference | Recall 0.296 → 0.549 (+86%) | Done |
| ✅ Done | Aggressive NMS + min area filter | Reduces FP from tile overlap | Done |
| ✅ Done | SORT tracker with Kalman filter | IDSW rate 1.6%, persistent IDs | Done |
| ✅ Done | ONNX export | 0.31 MB, opset 11, ONNXRuntime verified | Done |
| ✅ Done | Precision / Recall / F1 evaluation | Detection quality beyond counting | Done |
| ✅ Done | IDSW / MOTA tracking evaluation | Tracking stability quantified | Done |
| 🔴 Future | Fine-tune on CCTV transport dataset | Primary fix for MOTA and high-density recall | Planned |
| 🔴 Future | TensorRT / GPU deployment | Real-time enhanced mode (target: ≥ 10 FPS) | Planned |
| 🟡 Future | DeepSORT with appearance features | Robust re-ID after heavy occlusion | Planned |
| 🟡 Future | Virtual tripwire / line crossing | Accurate entry/exit per gate | Planned |
| 🟢 Future | Upgrade to YOLOv8m or YOLOv9 | Higher mAP on dense crowds | Planned |

---

## 7. Evaluation Reproducibility

```bash
# Standard evaluation
python -m src.evaluation.evaluate \
    --dataset data/mot20/train/MOT20-01 \
    --conf 0.4 --device cpu \
    --output assets/sample_outputs/eval_results.json \
    --save-samples assets/sample_outputs/

# Enhanced + Tracking evaluation
python -m src.evaluation.evaluate \
    --dataset data/mot20/train/MOT20-01 \
    --conf 0.3 --device cpu --enhance --track \
    --output assets/sample_outputs/eval_results_tracked.json \
    --save-samples assets/sample_outputs/

# ONNX export + verification
python -m src.inference.export_onnx \
    --output assets/yolov5s.onnx --verify
```

Artifact files committed at `assets/sample_outputs/eval_results.json`,
`eval_results_enhanced.json`, `eval_results_tracked.json`, and `yolov5s_metadata.json`.

---

## 8. Ethical Considerations

This repository is built for **aggregate capacity monitoring**, not individual surveillance. Deployment in public environments like TransJakarta must consider:

- Transparency to passengers that an automated counting system is active.
- No persistent storage of footage or bounding box data that could identify individuals.
- Compliance with applicable personal data protection regulations (UU PDP Indonesia).
- Independent accuracy validation before use in capacity or safety decisions.