<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv5s-COCO-00FFAB?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SORT-Kalman-FF6B6B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/ONNX-Exported-005CED?style=for-the-badge&logo=onnx&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/License-AGPL--3.0-blue?style=for-the-badge"/>

# 🚌 CCTV People Counting — YOLOv5

**People Detection, Counting & Tracking for TransJakarta BRT Station Monitoring**

End-to-end computer vision pipeline: YOLOv5s · CLAHE · tile-based inference · SORT + Kalman tracking · ONNX export · FastAPI · Docker

[📦 Quick Start](#quick-start-local) · [📊 Results](#key-results) · [🎯 Tracking](#tracking-mode) · [🐳 Docker](#quick-start-docker) · [🔍 Error Analysis](src/evaluation/error_analysis.md) · [📋 Model Card](MODEL_CARD.md)

</div>

---

## Why This Matters for TransJakarta

TransJakarta serves over **1 million daily trips** across hundreds of BRT shelters (halte) throughout Jakarta ([~1.4 million trips/day, ANTARA 2026](https://www.antaranews.com/berita/5361222/transjakarta-akui-ada-gap-antara-jumlah-penumpang-dan-infrastruktur)). Overcrowding at stations during peak hours creates safety risks and degrades service quality. This system provides automated passenger counting and tracking from CCTV feeds to:

- 🔴 **Monitor station capacity** — detect when platforms approach unsafe density levels
- 🚍 **Optimize fleet dispatch** — route additional buses to high-demand corridors
- 📱 **Improve passenger experience** — provide crowd estimates so commuters can plan routes

---

| Standard Mode | Enhanced Mode |
|---|---|
| ![Standard detection output](assets/sample_outputs/out_standard.jpg) | ![Enhanced detection output](assets/sample_outputs/out_enhanced.jpg) |
| Conf: 0.4 · Precision: 0.983 · Recall: 0.296 | Conf: 0.3 · CLAHE + Tile · Recall: 0.549 |

---

## Key Results

Evaluated on **MOT20-01** (429 frames, crowded pedestrian scenes, MOTChallenge ground truth):

### Counting Accuracy

| Metric | Standard Mode | Enhanced Mode | Δ |
|---|---|---|---|
| **MAE (per frame)** | 32.38 | **10.71** | ↓ 67% |
| **MAPE (per frame)** | 69.78% | **22.81%** | ↓ 67% |

### Detection Quality (IoU ≥ 0.5)

| Metric | Standard Mode | Enhanced Mode | Δ |
|---|---|---|---|
| **Precision** | **0.983** | 0.714 | trade-off |
| **Recall** | 0.296 | **0.549** | ↑ 86% |
| **F1 Score** | 0.455 | **0.621** | ↑ 37% |

### Tracking Quality (SORT + Kalman, Enhanced Mode)

| Metric | Value | Interpretation |
|---|---|---|
| **ID Switches (IDSW)** | 318 | Times a GT person changed assigned track ID |
| **IDSW Rate** | **1.6%** | Low = stable tracker across dense crowd |
| **MOTA** | −0.0323 | Expected for pretrained model on high-density benchmark — see [Model Card](MODEL_CARD.md) |

### Throughput

| Mode | Avg FPS | Confidence |
|---|---|---|
| Standard | 3.59 | 0.4 |
| Enhanced | 0.31 | 0.3 |
| Enhanced + SORT | ~0.31 | 0.3 |

> **Design choice:** Standard mode achieves near-perfect precision (98.3%) but misses ~70% of people in dense crowds. Enhanced mode trades precision for dramatically better recall (+86%), yielding 67% lower counting error. SORT tracker adds persistent IDs with only 1.6% ID switch rate — the right balance for crowded BRT monitoring.

<details>
<summary>📂 Results Provenance (click to expand)</summary>

- **Dataset:** MOT20-01 (MOTChallenge, downloaded from motchallenge.net)
- **Model:** YOLOv5s pretrained COCO via `torch.hub`
- **Hardware:** Intel Core CPU, Python 3.12
- **Standard run:**
  ```bash
  python -m src.evaluation.evaluate --dataset data/mot20/train/MOT20-01 \
    --conf 0.4 --device cpu --output assets/sample_outputs/eval_results.json
  ```
- **Enhanced run:**
  ```bash
  python -m src.evaluation.evaluate --dataset data/mot20/train/MOT20-01 \
    --conf 0.3 --device cpu --enhance \
    --output assets/sample_outputs/eval_results_enhanced.json
  ```
- **Tracking run:**
  ```bash
  python -m src.evaluation.evaluate --dataset data/mot20/train/MOT20-01 \
    --conf 0.3 --device cpu --enhance --track \
    --output assets/sample_outputs/eval_results_tracked.json
  ```
- **Artifact files:** [`eval_results.json`](assets/sample_outputs/eval_results.json) · [`eval_results_enhanced.json`](assets/sample_outputs/eval_results_enhanced.json) · [`eval_results_tracked.json`](assets/sample_outputs/eval_results_tracked.json)
- **Detection matching:** Greedy IoU matching, threshold ≥ 0.5

</details>

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection Model | YOLOv5s (pretrained COCO, via `torch.hub`) |
| Deep Learning | PyTorch ≥ 2.0 |
| Video I/O | OpenCV ≥ 4.8 |
| Image Preprocessing | Pillow (PIL) ≥ 10.0 · CLAHE (OpenCV) |
| Tracking | SORT with Kalman filter (`filterpy`) + IoU association |
| Model Export | ONNX (opset 11, 0.31 MB, ONNXRuntime verified) |
| API Framework | FastAPI + Uvicorn |
| Container | Docker |

---

## Quick Start (Local)

### 1. Setup

```bash
git clone https://github.com/azka-art/cctv-people-counting-yolov5.git
cd cctv-people-counting-yolov5

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements.txt
```

> **Note:** YOLOv5s weights (~14MB) auto-download on first run via `torch.hub`. Internet required for first execution only.

---

### 2. Image Inference

```bash
# Standard mode
python -m src.inference.inference_image \
    --input assets/sample.jpg \
    --output assets/sample_outputs/out_standard.jpg \
    --conf 0.4 --device cpu

# Enhanced mode — recommended for crowded/backlit scenes
python -m src.inference.inference_image \
    --input assets/sample.jpg \
    --output assets/sample_outputs/out_enhanced.jpg \
    --conf 0.3 --device cpu --enhance
```

**Expected output:** Annotated image with bounding boxes, confidence scores, and `People Count: N` overlay.

---

### 3. Video Inference

```bash
# Standard with tracking
python -m src.inference.inference_video \
    --input assets/demo_input.mp4 \
    --output assets/demo_output.mp4 \
    --conf 0.4 --device cpu --track

# Enhanced + SORT tracking — best accuracy for dense crowds
python -m src.inference.inference_video \
    --input assets/demo_input.mp4 \
    --output assets/demo_output.mp4 \
    --conf 0.3 --device cpu --enhance --track

# Custom Kalman age tolerance
python -m src.inference.inference_video \
    --input assets/demo_input.mp4 \
    --output assets/demo_output.mp4 \
    --conf 0.3 --device cpu --enhance --track --max-age 45
```

**Expected output:** `assets/demo_output.mp4` — annotated video with bbox + confidence + `People Count / Unique Total / ID Switches` overlay per frame.

---

### 4. ONNX Export

```bash
python -m src.inference.export_onnx \
    --output assets/yolov5s.onnx --verify
```

**Expected output:** `assets/yolov5s.onnx` (0.31 MB, opset 11) + `assets/yolov5s_metadata.json`

Deployment targets:
```bash
# ONNXRuntime
pip install onnxruntime

# TensorRT
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.trt

# OpenVINO
mo --input_model yolov5s.onnx
```

---

### 5. API Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

```bash
# Health check
curl http://localhost:8000/
# → {"status": "ok"}

# Standard detection
curl -X POST "http://localhost:8000/detect/image" \
    -F "file=@assets/sample.jpg"

# Enhanced detection
curl -X POST "http://localhost:8000/detect/image?enhance=true&conf=0.3" \
    -F "file=@assets/sample.jpg"
```

**Response contract:**

```json
{
  "count": 28,
  "detections": [
    {"x1": 15, "y1": 30, "x2": 110, "y2": 240, "score": 0.88},
    {"x1": 200, "y1": 45, "x2": 310, "y2": 250, "score": 0.76}
  ]
}
```

> 💡 Interactive API docs at `http://localhost:8000/docs` (Swagger UI)

---

## Quick Start (Docker)

```bash
docker build -t tj-cv-api -f docker/Dockerfile .
docker run -p 8000:8000 tj-cv-api
```

```bash
curl http://localhost:8000/
# → {"status": "ok"}

curl -X POST "http://localhost:8000/detect/image" \
    -F "file=@assets/sample.jpg"
```

---

## Cloud Deployment

| Platform | Command |
|---|---|
| **Google Cloud Run** | `gcloud run deploy --image <IMAGE> --port 8000` |
| **AWS ECS Fargate** | Task definition + service (CPU-only, no GPU required) |
| **Any Docker host** | `docker run -p 8000:8000 tj-cv-api` |

> **Cloud Run note:** Container binds to port 8000. Use `--port 8000` in `gcloud run deploy`, or modify Dockerfile CMD to bind to `$PORT` for native Cloud Run compatibility.

---

## Enhanced Mode — How It Works

### 1. CLAHE Preprocessing

**Contrast Limited Adaptive Histogram Equalization** normalizes local contrast to recover detail in dark and overexposed regions — mitigating false negatives from backlight and poor lighting ([Error Analysis Case 4](src/evaluation/error_analysis.md)).

### 2. Tile-Based Inference

Splits each frame into overlapping 640px tiles, runs detection on each tile, then merges results with aggressive NMS (IoU=0.3) and minimum box area filtering (1 500 px²) — catching small and distant people that full-image inference misses ([Error Analysis Cases 1 & 5](src/evaluation/error_analysis.md)).

| Technique | Failure Mode Addressed | Measured Impact |
|---|---|---|
| CLAHE | Backlight / low-light FN | Recovers silhouette detail |
| Tile inference | Small/distant people FN | Recall 0.296 → 0.549 (+86%) |
| Aggressive NMS | Duplicate boxes from tile overlap | Reduces FP from tile merging |
| Min box area filter | Noise/spurious detections | Removes boxes < 1 500 px² |

---

## Tracking Mode

The `--track` flag enables **SORT tracking with Kalman filter** for persistent ID assignment across frames:

- **Kalman filter prediction** — bridges detection gaps during brief occlusion
- **Persistent unique IDs** — color-coded per person across entire video
- **ID switch counting** — quantifies tracking stability (IDSW = 318, rate = 1.6% on MOT20-01)
- **Terminal summary** — IDSW rate + stability assessment printed at end of video

```bash
python -m src.inference.inference_video \
    --input assets/demo_input.mp4 \
    --output assets/demo_output_tracked.mp4 \
    --conf 0.3 --device cpu --enhance --track
```

**Per-frame overlay shows three metrics:**
```
People Count : 12   ← active detections this frame
Unique Total : 34   ← unique person IDs since video start
ID Switches  : 2    ← tracking stability indicator
```

> For production use cases requiring robust re-identification after prolonged occlusion, consider upgrading to DeepSORT or ByteTrack.

---

## Evaluation

Requires MOT20 dataset. See [DATA_SOURCES.md](DATA_SOURCES.md) for download instructions.

```bash
# Standard
python -m src.evaluation.evaluate \
    --dataset data/mot20/train/MOT20-01 \
    --conf 0.4 --device cpu \
    --output assets/sample_outputs/eval_results.json \
    --save-samples assets/sample_outputs/

# Enhanced
python -m src.evaluation.evaluate \
    --dataset data/mot20/train/MOT20-01 \
    --conf 0.3 --device cpu --enhance \
    --output assets/sample_outputs/eval_results_enhanced.json \
    --save-samples assets/sample_outputs/

# Enhanced + Tracking (IDSW / MOTA)
python -m src.evaluation.evaluate \
    --dataset data/mot20/train/MOT20-01 \
    --conf 0.3 --device cpu --enhance --track \
    --output assets/sample_outputs/eval_results_tracked.json \
    --save-samples assets/sample_outputs/
```

Computes: **MAE/MAPE** (counting) · **Precision/Recall/F1** (IoU ≥ 0.5) · **IDSW/MOTA** (tracking, when `--track` enabled).

---

## Error Analysis

Documented in [`src/evaluation/error_analysis.md`](src/evaluation/error_analysis.md) — 5 concrete failure cases:

| # | Type | Condition | Error | Mitigation |
|---|---|---|---|---|
| 1 | **False Negative** | Severe occlusion (>70% body covered) | −4 per frame | Tile inference |
| 2 | **False Negative** | Motion blur (≥ 2 m/s) | −2 per frame | Temporal averaging |
| 3 | **False Positive** | Human figures in ads/posters | +2 per frame | ROI masking, min area filter |
| 4 | **False Negative** | Backlight / low illumination | −5 per frame | CLAHE preprocessing |
| 5 | **False Negative** | High density + bird-eye distance | −11 per frame | Tile inference |

---

## Project Structure

```
cctv-people-counting-yolov5/
├── README.md
├── MODEL_CARD.md                  # Model details, metrics, ethical considerations
├── DATA_SOURCES.md                # Dataset provenance and download instructions
├── requirements.txt
├── .gitignore
├── src/
│   ├── inference/
│   │   ├── inference_image.py     # Image detection (PIL + YOLOv5s)
│   │   ├── inference_video.py     # Video detection + tracking overlay
│   │   ├── enhance.py             # CLAHE + tile-based inference
│   │   ├── tracker.py             # SORT tracker with Kalman filter
│   │   ├── export_onnx.py         # ONNX export + ONNXRuntime verification
│   │   └── visualize.py           # Visualization utilities
│   ├── api/
│   │   ├── app.py                 # FastAPI application
│   │   └── schemas.py             # Pydantic response models
│   └── evaluation/
│       ├── evaluate.py            # MAE/MAPE + P/R/F1 + IDSW/MOTA evaluation
│       └── error_analysis.md      # 5 FP/FN failure cases with mitigations
├── docker/
│   └── Dockerfile
└── assets/
    ├── demo_input.mp4             # Demo input video
    ├── demo_output.mp4            # Demo output (annotated)
    ├── yolov5s.onnx               # Exported ONNX model (0.31 MB)
    ├── yolov5s_metadata.json      # Export provenance
    └── sample_outputs/            # Eval JSON, screenshots, error frames
```

---

## Limitations

- **Tracking uses IoU + Kalman only** — no appearance features; may lose tracks during prolonged heavy occlusion.
- **Pretrained model** — enhanced mode improves recall to 0.55 but still misses ~45% of people in the densest crowd conditions.
- **Enhanced mode FPS** — ~0.31 FPS on CPU. GPU + TensorRT required for real-time deployment.
- **MOTA is negative** — expected for a non-fine-tuned model on high-density MOT20 data. See [Model Card](MODEL_CARD.md) for full explanation.
- **Not production-grade** — portfolio demonstration. Production requires fine-tuning on domain-specific CCTV data.

---

## Future Improvements

| Priority | Improvement | Expected Impact |
|---|---|---|
| 🔴 High | Fine-tune on CCTV transport dataset | Primary fix for recall and MOTA on domain data |
| 🔴 High | TensorRT / GPU deployment | Real-time enhanced mode (target: ≥ 10 FPS) |
| 🟡 Medium | DeepSORT / ByteTrack with appearance features | Robust re-ID, lower IDSW after heavy occlusion |
| 🟡 Medium | Virtual tripwire / line crossing counter | Accurate entry/exit per gate |
| 🟢 Low | Upgrade to YOLOv8m or YOLOv9 | Higher mAP on dense crowds |

---

## Reproducibility

- **Model weights:** Auto-downloaded via `torch.hub.load('ultralytics/yolov5', 'yolov5s')` — not committed to Git
- **ONNX model:** `assets/yolov5s.onnx` (opset 11, 0.31 MB) — committed to repo
- **Docker:** Weights pre-downloaded at build time for offline inference
- **Eval artifacts:** `assets/sample_outputs/eval_results*.json` committed with exact run parameters
- **Python:** 3.12 (tested), 3.9+ compatible

---

## License

Portfolio and educational use. Uses [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) under [AGPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE). Network service deployment requires AGPL-3.0 compliance or Ultralytics Enterprise licensing.

---

<div align="center">

Built with ❤️ for TransJakarta BRT · Jakarta, Indonesia

</div>