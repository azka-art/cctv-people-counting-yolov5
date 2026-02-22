# CCTV People Counting — YOLOv5

CCTV-like **People Detection & Counting** system built for monitoring passenger density at TransJakarta BRT stations.

---

## Why This Matters for TransJakarta

TransJakarta operates 250+ halte (stations) serving 1M+ daily passengers. Overcrowding at stations during peak hours creates safety risks and degrades service quality. This system provides automated passenger counting from CCTV feeds to:

- **Monitor station capacity** — detect when platforms approach unsafe density levels
- **Optimize fleet dispatch** — route additional buses to high-demand corridors
- **Improve passenger experience** — provide crowd estimates so commuters can plan routes

---

## Demo

| Input | Output |
|---|---|
| `assets/demo_input.mp4` | `assets/demo_output.mp4` |

Output video contains: bounding boxes per person, confidence scores, and **People Count** overlay on every frame.

---

## Key Results

Evaluated on MOT20-01 dataset (crowded pedestrian scenes with ground truth annotations):

| Metric | Standard Mode | Enhanced Mode | Improvement |
|---|---|---|---|
| **MAE (per frame)** | 32.38 | **7.0** | ↓ 78% |
| **MAPE (per frame)** | 69.78% | **19.13%** | ↓ 72% |
| **Avg FPS** | 3.64 | 0.35 | Trade-off |
| **Confidence** | 0.4 | 0.3 | — |
| **Frames Evaluated** | 429 | 50 | — |

> **Enhanced mode** uses CLAHE preprocessing + tile-based inference to dramatically improve detection in crowded scenes. See [Enhancement Details](#enhanced-mode) below.

Hardware: Intel Core CPU | Device: CPU | Dataset: MOT20-01

---

## Tech Stack

| Component | Technology |
|---|---|
| Detection Model | YOLOv5s (pretrained COCO, via `torch.hub`) |
| Deep Learning | PyTorch ≥ 2.0 |
| Video Processing | OpenCV ≥ 4.8 |
| Image I/O | Pillow (PIL) ≥ 10.0 |
| API Framework | FastAPI + Uvicorn |
| Container | Docker |

---

## Quick Start (Local)

### 1. Setup

```bash
git clone https://github.com/<your-username>/cctv-people-counting-yolov5.git
cd cctv-people-counting-yolov5

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements.txt
```

> **Note:** YOLOv5s weights auto-download on first run (~14MB). Internet access required on first execution.

### 2. Image Inference

```bash
# Standard mode
python -m src.inference.inference_image \
    --input assets/sample.jpg \
    --output assets/sample_outputs/out_standard.jpg \
    --conf 0.4 --device cpu

# Enhanced mode (recommended for crowded scenes)
python -m src.inference.inference_image \
    --input assets/sample.jpg \
    --output assets/sample_outputs/out_enhanced.jpg \
    --conf 0.3 --device cpu --enhance
```

### 3. Video Inference

```bash
python -m src.inference.inference_video \
    --input assets/demo_input.mp4 \
    --output assets/demo_output.mp4 \
    --conf 0.3 --device cpu --enhance
```

### 4. API Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Test:
```bash
# Health check
curl http://localhost:8000/

# Standard detection
curl -X POST "http://localhost:8000/detect/image" \
    -F "file=@assets/sample.jpg"

# Enhanced detection
curl -X POST "http://localhost:8000/detect/image?enhance=true&conf=0.3" \
    -F "file=@assets/sample.jpg"
```

**Response:**
```json
{
  "count": 28,
  "detections": [
    {"x1": 15, "y1": 30, "x2": 110, "y2": 240, "score": 0.88},
    {"x1": 200, "y1": 45, "x2": 310, "y2": 250, "score": 0.76}
  ]
}
```

> **Tip:** Interactive API docs at `http://localhost:8000/docs` (Swagger UI).

---

## Quick Start (Docker)

```bash
docker build -t tj-cv-api -f docker/Dockerfile .
docker run -p 8000:8000 tj-cv-api
```

Verify:
```bash
curl http://localhost:8000/
# {"status": "ok"}

curl -X POST "http://localhost:8000/detect/image" -F "file=@assets/sample.jpg"
```

---

## Enhanced Mode

The system implements two enhancement techniques that directly address documented failure cases in crowded transit environments:

### 1. CLAHE Preprocessing
**Contrast Limited Adaptive Histogram Equalization** normalizes local contrast to recover detail in dark/overexposed regions. This mitigates false negatives caused by backlight and poor lighting conditions (Error Analysis Case 4).

### 2. Tile-Based Inference
Splits the frame into overlapping 640px tiles, runs detection on each tile, then merges results with aggressive NMS (IoU=0.3) and minimum box area filtering (1500px²). This catches small/distant people that full-image inference misses (Error Analysis Cases 1 & 5).

| Technique | Mitigates | Impact |
|---|---|---|
| CLAHE | Backlight / low-light FN | Recovers silhouette detail |
| Tile inference | Small/distant people FN | 5→28 detections on crowded test image |
| Aggressive NMS | Duplicate boxes from tiles | Reduces FP from tile overlap |
| Min box area filter | Noise/spurious detections | Removes boxes < 1500px² |

Enable with `--enhance` flag on any CLI command or `?enhance=true` API parameter.

---

## Evaluation

Requires MOT20 dataset. See [DATA_SOURCES.md](DATA_SOURCES.md) for download instructions.

```bash
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
    --save-samples assets/sample_outputs/ --max-frames 50
```

> See [MODEL_CARD.md](MODEL_CARD.md) for full model details and performance metrics.

---

## Error Analysis

Documented in [`src/evaluation/error_analysis.md`](src/evaluation/error_analysis.md).

5 concrete failure cases with root cause analysis and mitigations:

1. **Severe Occlusion** — undercount when passengers overlap (FN) → mitigated by tile inference
2. **Motion Blur** — missed detections on fast-moving people (FN)
3. **Poster/Ads** — human figures in ads detected as real people (FP) → mitigated by min area filter
4. **Backlight/Low Light** — silhouettes not detected (FN) → mitigated by CLAHE
5. **High Density + Distance** — small distant people missed (FN) → mitigated by tile inference

---

## Project Structure

```
├── README.md                         # This file
├── MODEL_CARD.md                     # Model details, metrics, limitations
├── DATA_SOURCES.md                   # Dataset sources and download instructions
├── requirements.txt                  # Python dependencies
├── .gitignore
├── src/
│   ├── inference/
│   │   ├── inference_image.py        # Image detection (PIL + YOLOv5s)
│   │   ├── inference_video.py        # Video detection (OpenCV + YOLOv5s)
│   │   ├── enhance.py               # CLAHE + tile-based inference
│   │   └── visualize.py             # Visualization helper
│   ├── api/
│   │   ├── app.py                    # FastAPI application
│   │   └── schemas.py               # Pydantic response models
│   └── evaluation/
│       ├── evaluate.py              # MAE/MAPE evaluation script
│       └── error_analysis.md        # 5+ FP/FN failure cases
├── docker/
│   └── Dockerfile
└── assets/
    ├── demo_input.mp4               # Demo input video
    ├── demo_output.mp4              # Demo output (annotated)
    └── sample_outputs/              # Sample annotated images
```

---

## Limitations

- **No tracking** — Each frame is independent; the same person may be counted multiple times across frames. Future work: SORT/DeepSORT integration.
- **Pretrained model** — YOLOv5s trained on COCO may underperform on extreme occlusion or unusual angles without fine-tuning.
- **Enhanced mode FPS** — Tile-based inference runs at ~0.35 FPS on CPU. GPU acceleration or model optimization needed for real-time use.
- **Not production-grade** — This is a portfolio demonstration. Production deployment requires fine-tuning on domain data, tracking, and hardware optimization.

See [Error Analysis](src/evaluation/error_analysis.md) for detailed failure mode documentation.

---

## Future Improvements

| Priority | Improvement | Impact |
|---|---|---|
| High | SORT/DeepSORT tracking | Eliminate double-counting, enable flow counting |
| High | Fine-tune on CCTV transport dataset | Better accuracy for halte conditions |
| High | GPU deployment | Real-time enhanced mode inference |
| Medium | Virtual tripwire line crossing | Accurate entry/exit counting |
| Low | Upgrade to YOLOv8m/YOLOv9 | Higher mAP on dense crowds |

---

## License

This project is for portfolio/educational purposes. Model weights (YOLOv5s) follow the [Ultralytics AGPL-3.0 License](https://github.com/ultralytics/yolov5/blob/master/LICENSE).
