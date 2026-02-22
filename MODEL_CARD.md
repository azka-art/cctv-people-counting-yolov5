# Model Card — People Detection & Counting

> **Versi:** 1.0 | **Tanggal:** 2025 | **Framework:** PyTorch (YOLOv5s via `torch.hub`)

---

## 1. Model Overview

| Atribut | Detail |
|---|---|
| **Task** | Object Detection + Per-frame People Counting |
| **Architecture** | YOLOv5s (Small variant — efisien, latency rendah) |
| **Pretrained On** | COCO 2017 (80 kelas, termasuk `person` → Class 0) |
| **Model Source** | `torch.hub.load('ultralytics/yolov5', 'yolov5s')` |
| **Output** | Bounding boxes, confidence scores, People Count overlay |
| **Target Class** | `person` saja (kelas lain difilter di post-processing) |
| **Default Confidence Threshold** | `0.4` (standard) / `0.3` (enhanced) |

---

## 2. Intended Use

### 2.1 Skenario yang Sesuai

- **Pemantauan kepadatan halte BRT:** Menghitung jumlah orang di area tunggu atau koridor secara agregat per frame.
- **Analisis throughput antrean:** Mengukur pergerakan penumpang pada kamera statis sudut elevasi tinggi.
- **Demo pipeline end-to-end:** Menunjukkan integrasi Computer Vision ke layanan REST API dalam satu repository.
- **Evaluasi kapasitas armada:** Sebagai sinyal awal load monitoring pada halte dengan volume penumpang tinggi.

### 2.2 Skenario yang TIDAK Sesuai (Out-of-Scope)

- ❌ Identifikasi biometrik atau face recognition individu.
- ❌ Pelacakan identitas untuk keperluan penegakan hukum atau keamanan.
- ❌ Pelaporan statistik resmi tanpa kalibrasi spasial dan validasi manual.
- ❌ Deployment produksi kritis tanpa audit bias dan akurasi tambahan pada data domain.

---

## 3. Pipeline Inference

### 3.1 Standard Mode

| Langkah | Tools | Keterangan |
|---|---|---|
| Load gambar | `PIL.Image.open()` → konversi RGB | Digunakan di `inference_image.py` |
| Load frame video | `cv2.VideoCapture` → BGR ke RGB | Digunakan di `inference_video.py` |
| Resize otomatis | YOLOv5 internal (letterbox) | Model menangani resize ke 640×640 |
| Filter class | Post-processing | Hanya class 0 (person) |

### 3.2 Enhanced Mode

| Langkah | Tools | Keterangan |
|---|---|---|
| CLAHE preprocessing | `cv2.createCLAHE` pada L-channel (LAB) | Normalisasi kontras lokal |
| Tile-based inference | 640px tiles, 25% overlap | Deteksi orang kecil/jauh |
| Aggressive NMS | IoU threshold 0.3 | Hapus duplikat dari tile overlap |
| Min area filter | 1500px² minimum | Hapus deteksi noise/spurious |

### 3.3 Output Format — API Contract

Endpoint `POST /detect/image` mengembalikan JSON dengan struktur **persis** berikut:

```json
{
  "count": 28,
  "detections": [
    { "x1": 15,  "y1": 30,  "x2": 110, "y2": 240, "score": 0.88 },
    { "x1": 200, "y1": 45,  "x2": 310, "y2": 250, "score": 0.76 },
    { "x1": 400, "y1": 60,  "x2": 490, "y2": 235, "score": 0.61 }
  ]
}
```

> Koordinat dalam piksel relatif terhadap resolusi gambar input. `score` adalah confidence value [0.0–1.0].

---

## 4. Performance Metrics

### 4.1 Standard Mode (conf=0.4)

| Metrik | Nilai |
|---|---|
| **Dataset Evaluasi** | MOT20-01 (429 frames) |
| **Confidence Threshold** | 0.4 |
| **MAE (Mean Absolute Error)** | 32.38 |
| **MAPE (Mean Absolute Percentage Error)** | 69.78% |
| **Inference Speed (FPS)** | 3.64 |
| **Overcount frames** | 0 |
| **Undercount frames** | 429 (100%) |
| **Hardware** | Intel Core CPU |
| **Device** | CPU |

### 4.2 Enhanced Mode (CLAHE + Tile, conf=0.3)

| Metrik | Nilai |
|---|---|
| **Dataset Evaluasi** | MOT20-01 (50 frames) |
| **Confidence Threshold** | 0.3 |
| **MAE (Mean Absolute Error)** | **7.0** |
| **MAPE (Mean Absolute Percentage Error)** | **19.13%** |
| **Inference Speed (FPS)** | 0.35 |
| **Overcount frames** | 0 |
| **Undercount frames** | 50 (100%) |
| **Hardware** | Intel Core CPU |
| **Device** | CPU |

### 4.3 Improvement Summary

| Metrik | Standard → Enhanced | Change |
|---|---|---|
| MAE | 32.38 → 7.0 | **↓ 78%** |
| MAPE | 69.78% → 19.13% | **↓ 72%** |
| FPS | 3.64 → 0.35 | ↓ 90% (trade-off) |

> Enhanced mode FPS dapat ditingkatkan signifikan dengan GPU acceleration. Tile inference bersifat parallelizable.

**Cara menjalankan evaluasi:**
```bash
# Standard
python -m src.evaluation.evaluate \
    --dataset data/mot20/train/MOT20-01 \
    --conf 0.4 --device cpu \
    --output assets/sample_outputs/eval_results.json

# Enhanced
python -m src.evaluation.evaluate \
    --dataset data/mot20/train/MOT20-01 \
    --conf 0.3 --device cpu --enhance \
    --output assets/sample_outputs/eval_results_enhanced.json \
    --max-frames 50
```

---

## 5. Limitations

Sistem deteksi berbasis YOLOv5s pretrained COCO memiliki keterbatasan inheren yang perlu dipahami sebelum deployment. Detail lengkap ada di [`src/evaluation/error_analysis.md`](../src/evaluation/error_analysis.md).

**Failure modes utama:**

| # | Tipe | Kondisi | Dampak | Mitigasi |
|---|---|---|---|---|
| 1 | **False Negative** | Severe occlusion (>70% tubuh tertutup) | Undercount pada kondisi halte crush load | ✅ Tile inference |
| 2 | **False Negative** | Motion blur frame (kecepatan ≥ 2 m/s) | Momen sibuk saat pintu bus terbuka | — |
| 3 | **False Positive** | Poster/iklan figur manusia di dinding | Overcount pada area dengan visual display | ✅ Min area filter |
| 4 | **False Negative** | Pencahayaan buruk (malam / backlight) | Deteksi gagal di halte minim penerangan | ✅ CLAHE |
| 5 | **False Negative** | Kepadatan tinggi + perspektif jauh | Undercount 50% pada kamera elevated | ✅ Tile inference |

---

## 6. Mitigations & Future Work

| Prioritas | Langkah | Dampak | Status |
|---|---|---|---|
| **Tinggi** | CLAHE preprocessing | Mengurangi FN pada kondisi backlight/gelap | ✅ Implemented |
| **Tinggi** | Tile-based inference | Deteksi orang kecil/jauh, mengurangi undercount | ✅ Implemented |
| **Tinggi** | Aggressive NMS + min area filter | Mengurangi FP dari tile overlap dan noise | ✅ Implemented |
| **Tinggi** | SORT/DeepSORT tracking | Eliminasi double-count, hitung flow masuk/keluar | ⬜ Future |
| **Menengah** | Fine-tuning pada dataset CCTV transportasi | Akurasi lebih baik untuk kondisi halte Indonesia | ⬜ Future |
| **Menengah** | Virtual tripwire / line crossing counter | Counting akurat untuk pintu masuk-keluar halte | ⬜ Future |
| **Rendah** | Model upgrade ke YOLOv8m atau YOLOv9 | Peningkatan mAP pada kepadatan tinggi | ⬜ Future |

---

## 7. Ethical Considerations

Repository ini dibangun untuk tujuan **monitoring agregat kapasitas**, bukan surveillance individu. Implementasi sistem ini di lingkungan publik seperti TransJakarta harus mempertimbangkan:

- Transparansi kepada penumpang bahwa sistem counting aktif beroperasi.
- Tidak menyimpan footage atau bounding box yang dapat mengidentifikasi individu.
- Compliance dengan regulasi perlindungan data pribadi yang berlaku.
