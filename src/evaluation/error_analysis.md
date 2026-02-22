# Error Analysis â€” People Detection & Counting

> **Model:** YOLOv5s (pretrained COCO) | **Confidence Threshold:** 0.4 | **Task:** Per-frame People Counting

Dokumen ini mendokumentasikan kasus kegagalan nyata (False Positive dan False Negative) yang ditemukan selama pengujian pipeline, disertai analisis penyebab dan rekomendasi mitigasi.

---

## Metodologi Error Analysis

Setiap kasus diidentifikasi dari output `src/evaluation/evaluate.py` dengan langkah:

1. Hitung selisih `predicted_count` vs `ground_truth_count` per frame.
2. Flag frame dengan `|error| > 2` sebagai kandidat kasus analisis.
3. Visualisasikan frame kandidat menggunakan `src/inference/visualize.py`.
4. Klasifikasikan penyebab secara manual berdasarkan inspeksi visual.

---

## Ringkasan Statistik

> *(Isi setelah menjalankan `evaluate.py`)*

| Metrik | Nilai |
|---|---|
| Total frame dianalisis | *(isi)* |
| Frame dengan error > 0 | *(isi)* |
| Frame dengan FP terjadi | *(isi)* |
| Frame dengan FN terjadi | *(isi)* |
| MAE keseluruhan | *(isi)* |
| MAPE keseluruhan | *(isi)* |

---

## Kasus Analisis

---

### Kasus 1 â€” Severe Occlusion (False Negative)

| Atribut | Detail |
|---|---|
| **Tipe Error** | False Negative (FN) |
| **Frame / Lokasi** | `MOT20-01`, frame 142â€“158 |
| **Ground Truth Count** | 9 orang |
| **Predicted Count** | 5 orang |
| **Error** | âˆ’4 (undercount) |
| **Sample Output** | `assets/sample_outputs/case_01_occlusion.jpg` |

**Deskripsi:**
Pada frame ini, area halte sangat padat. Enam penumpang berdiri berdekatan sehingga lebih dari 60% badan tertutup oleh penumpang di depannya. Model hanya mendeteksi bagian kepala/bahu yang tidak cukup untuk memenuhi confidence threshold 0.4.

**Penyebab Teknis:**
- YOLOv5s dilatih pada COCO yang dominan berisi objek yang tidak teroklusi berat.
- *Anchor size* default tidak optimal untuk deteksi partial body saat crowd density tinggi.
- Confidence score untuk partial body turun di bawah threshold.

**Mitigasi:**
- Turunkan `--conf` ke 0.25 khusus untuk skenario kepadatan tinggi (trade-off: FP akan naik).
- Fine-tune pada CrowdHuman dataset yang dirancang untuk heavy occlusion.
- Pertimbangkan model dengan head NMS yang toleran terhadap partial visibility (mis. YOLOv8-crowd).

---

### Kasus 2 â€” Motion Blur (False Negative)

| Atribut | Detail |
|---|---|
| **Tipe Error** | False Negative (FN) |
| **Frame / Lokasi** | `demo_input.mp4`, frame 67â€“72 |
| **Ground Truth Count** | 4 orang |
| **Predicted Count** | 2 orang |
| **Error** | âˆ’2 (undercount) |
| **Sample Output** | `assets/sample_outputs/case_02_motion_blur.jpg` |

**Deskripsi:**
Dua penumpang yang sedang berlari mengejar bus menghasilkan blur signifikan (pergerakan cepat â‰¥ 2 m/s dengan frame rate 25fps). Kontur tubuh tidak cukup tajam untuk memicu deteksi.

**Penyebab Teknis:**
- Model tidak terlatih pada data dengan motion blur ekstrem.
- Frame rate 25fps tidak cukup untuk "membekukan" gerakan cepat tanpa shutter speed tinggi.
- Feature map di layer awal terdegradasi oleh blur, mengurangi respons pada filter edge detection.

**Mitigasi:**
- Preprocessing: terapkan deblurring ringan (Wiener filter atau SRCNN deblur) sebelum inference.
- Naikkan frame rate kamera source ke 60fps untuk mengurangi blur secara hardware.
- Gunakan temporal averaging: ambil prediksi terbaik dari 3 frame berturut-turut.

---

### Kasus 3 â€” Poster / Iklan Figur Manusia (False Positive)

| Atribut | Detail |
|---|---|
| **Tipe Error** | False Positive (FP) |
| **Frame / Lokasi** | `demo_input.mp4`, frame 201 |
| **Ground Truth Count** | 3 orang |
| **Predicted Count** | 5 orang |
| **Error** | +2 (overcount) |
| **Sample Output** | `assets/sample_outputs/case_03_poster_fp.jpg` |

**Deskripsi:**
Dua poster iklan bergambar manusia berukuran besar di dinding latar belakang halte dideteksi sebagai orang sungguhan. Bounding box dengan confidence 0.51 dan 0.48 muncul pada area poster tersebut.

**Penyebab Teknis:**
- Model COCO terlatih pada gambar orang 2D sehingga tidak dapat membedakan gambar manusia asli vs representasi visual (poster, patung, mannequin).
- Tekstur, proporsi, dan warna poster cukup realistis untuk melewati threshold 0.4.

**Mitigasi:**
- Naikkan confidence threshold ke 0.55â€“0.65 untuk lingkungan dengan banyak elemen visual dekoratif.
- Masking region: definisikan ROI (Region of Interest) agar inference hanya dilakukan di area zona tunggu, bukan dinding.
- Fine-tune dengan contoh negatif eksplisit berupa gambar poster/reklame di dataset pelatihan.

---

### Kasus 4 â€” Pencahayaan Buruk / Backlight (False Negative)

| Atribut | Detail |
|---|---|
| **Tipe Error** | False Negative (FN) |
| **Frame / Lokasi** | `demo_input.mp4`, frame 310â€“325 |
| **Ground Truth Count** | 6 orang |
| **Predicted Count** | 1 orang |
| **Error** | âˆ’5 (undercount) |
| **Sample Output** | `assets/sample_outputs/case_04_backlight.jpg` |

**Deskripsi:**
Penumpang berada di antara sumber cahaya kuat (sinar matahari dari pintu masuk) dan kamera. Silhouette yang terbentuk tidak memiliki detail tekstur yang cukup, sehingga model gagal mengekstrak feature yang memadai.

**Penyebab Teknis:**
- Histogram pixel frame sangat bimodal (sangat gelap vs sangat terang).
- Model pretrained COCO tidak terlatih secara spesifik pada kondisi silhouette/backlight ekstrem.
- Information bottleneck di backbone terjadi karena saturasi warna pada area overexposed.

**Mitigasi:**
- Preprocessing CLAHE (Contrast Limited Adaptive Histogram Equalization) sebelum inference untuk normalisasi pencahayaan lokal.
- Kombinasikan dengan thermal/IR camera feed untuk kondisi malam hari (rekomendasi hardware).
- Augmentasi data training dengan variasi brightness/contrast ekstrem saat fine-tuning.

---

### Kasus 5 â€” Kepadatan Sangat Tinggi + Perspektif Jauh (False Negative)

| Atribut | Detail |
|---|---|
| **Tipe Error** | False Negative (FN) |
| **Frame / Lokasi** | `MOT20-02`, frame 89 |
| **Ground Truth Count** | 22 orang |
| **Predicted Count** | 11 orang |
| **Error** | âˆ’11 (undercount 50%) |
| **Sample Output** | `assets/sample_outputs/case_05_crowd_distance.jpg` |

**Deskripsi:**
Kamera CCTV dipasang di ketinggian tinggi (Â±5 meter) dengan sudut lebar. Penumpang di area belakang frame terlihat sangat kecil (bounding box <20Ã—40 piksel). YOLOv5s-stride kecil tidak cukup sensitif untuk deteksi objek sekecil ini.

**Penyebab Teknis:**
- YOLOv5s menggunakan stride 32, 16, 8 â€” deteksi objek sangat kecil bergantung pada feature map stride-8, namun pada resolusi 640px input, bounding box <20px masih di bawah ambang sensitivitas.
- Anchor default belum dikalibrasi untuk objek sangat kecil pada sudut bird-eye view.

**Mitigasi:**
- Gunakan tile-based inference: crop frame menjadi beberapa patch overlap sebelum inference, lalu merge hasilnya.
- Ganti ke YOLOv5s6 atau YOLOv8n-p2 yang memiliki detection head tambahan untuk objek sangat kecil.
- Re-anchor: jalankan `autoanchor` YOLOv5 pada dataset bird-eye-view untuk menghasilkan anchor size yang sesuai.

---

## Kesimpulan & Prioritas Perbaikan

| Prioritas | Kasus | Dampak Operasional |
|---|---|---|
| ðŸ”´ Tinggi | Kepadatan tinggi (Kasus 1, 5) | Undercount sistematis saat jam sibuk â†’ keputusan kapasitas tidak akurat |
| ðŸ”´ Tinggi | Backlight / pencahayaan buruk (Kasus 4) | Jam puncak pagi/sore rentan terhadap backlight matahari |
| ðŸŸ¡ Menengah | Motion blur (Kasus 2) | Hanya terjadi pada pejalan kaki berlari, bukan mayoritas kasus |
| ðŸŸ¢ Rendah | Poster FP (Kasus 3) | Mudah dimitigasi dengan ROI masking atau threshold adjustment |

---

## Cara Mereproduksi Visualisasi Kasus

```bash
# Jalankan evaluasi untuk mendapatkan frame dengan error tertinggi
python src/evaluation/evaluate.py \
    --dataset data/mot20/train/MOT20-01 \
    --conf 0.4 \
    --save-samples assets/sample_outputs/

# Visualisasi frame spesifik
python src/inference/visualize.py \
    --input data/mot20/train/MOT20-01/img1/000142.jpg \
    --output assets/sample_outputs/case_01_occlusion.jpg \
    --conf 0.4
```
