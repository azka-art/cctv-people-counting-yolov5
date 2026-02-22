# Data Sources â€” Porto: AI Engineer (Computer Vision) TransJakarta

Dokumen ini menjelaskan asal-usul seluruh data yang digunakan dalam repository ini, mencakup video demo, dataset evaluasi kuantitatif (dengan ground truth), dan material untuk *error analysis*.

---

## Prinsip Pengelolaan Data

| Prinsip | Keterangan |
|---|---|
| **Tidak di-commit** | Dataset besar atau berlisensi riset tidak pernah masuk ke Git history |
| **Reproducible** | Setiap dataset dapat diunduh ulang menggunakan instruksi di dokumen ini |
| **License-compliant** | Hanya aset dengan lisensi publik/CC0/CC-BY yang disimpan langsung di repo |

---

## 1. Demo Assets (Committed to Git)

### `assets/demo_input.mp4`

| Atribut | Detail |
|---|---|
| **Tujuan** | Input untuk menghasilkan `assets/demo_output.mp4` (annotated: bbox + confidence + People Count overlay) |
| **Durasi** | 10â€“30 detik (direkomendasikan) |
| **Resolusi** | Minimum 720p agar deteksi akurat |
| **Skenario** | Kondisi halte transit, pedestrian crossing, atau area antrean â€” merepresentasikan konteks operasional TransJakarta |
| **Sumber** | [Pexels Videos](https://www.pexels.com/) (kata kunci: `crowd walking`, `bus station pedestrians`, `station crowd`) |
| **Lisensi** | Pexels License (bebas royalti, tidak perlu atribusi) |

**Cara unduh:**
```bash
# Pilih video dari Pexels, unduh resolusi 1080p atau 720p
# Rename menjadi demo_input.mp4 dan letakkan di folder assets/
```

---

## 2. Evaluation Dataset (NOT Committed â€” Ground Truth Required)

Dataset berikut dibutuhkan untuk menjalankan `src/evaluation/evaluate.py` dan menghasilkan metrik MAE/MAPE.

### 2.1 MOTChallenge â€” MOT20 *(Rekomendasi Utama)*

Dataset densitas tinggi yang paling relevan dengan skenario kepadatan penumpang di halte BRT.

| Atribut | Detail |
|---|---|
| **URL** | https://motchallenge.net/data/MOT20/ |
| **Alasan dipilih** | Kepadatan crowd tinggi, anotasi per-frame akurat, umum digunakan sebagai benchmark |
| **Format anotasi** | `frame, id, x, y, w, h, conf, cls, vis` (MOT format) |
| **Lisensi** | Akademik / non-komersial |

**Struktur folder lokal setelah diunduh:**
```
data/
â””â”€â”€ mot20/
    â”œâ”€â”€ train/
    â”‚   â”œâ”€â”€ MOT20-01/
    â”‚   â”‚   â”œâ”€â”€ img1/          # Frame-frame video (*.jpg)
    â”‚   â”‚   â””â”€â”€ gt/
    â”‚   â”‚       â””â”€â”€ gt.txt     # Ground truth annotations
    â”‚   â””â”€â”€ MOT20-02/
    â””â”€â”€ test/
```

**Cara unduh:**
```bash
# 1. Daftar akun di motchallenge.net
# 2. Unduh MOT20.zip dari halaman dataset
# 3. Ekstrak ke direktori data/mot20/
unzip MOT20.zip -d data/mot20/
```

---

### 2.2 UCSD Pedestrian Dataset *(Alternatif)*

Cocok untuk kamera statis (angle tetap) yang mensimulasikan sudut pandang CCTV halte.

| Atribut | Detail |
|---|---|
| **URL** | https://visal.cs.cityu.edu.hk/downloads/ucsdpeds-vids/ |
| **Alasan dipilih** | Sudut kamera statis (CCTV-like), cocok untuk flow analysis dan counting |
| **Lisensi** | Akademik |

---

## 3. Error Analysis Dataset (Opsional)

Digunakan untuk memperkaya skenario FP/FN di `src/evaluation/error_analysis.md`.

### 3.1 CrowdHuman

| Atribut | Detail |
|---|---|
| **URL** | https://www.crowdhuman.org/ |
| **Alasan dipilih** | Dirancang khusus untuk menguji ketahanan model terhadap occlusion ekstrem |
| **Relevansi** | Mensimulasikan kondisi penumpang saling menutupi saat jam sibuk |
| **Lisensi** | Akademik / non-komersial |

---

## 4. Kebijakan .gitignore

File berikut **wajib** masuk ke `.gitignore` dan tidak boleh ter-commit:

```gitignore
# Dataset mentah
data/

# Model weights
*.pt
*.pth
*.onnx
*.weights

# Python cache
__pycache__/
*.pyc
.venv/
*.egg-info/

# Output sementara (bukan demo final)
outputs/
runs/
```

File yang **boleh** di-commit:
```
assets/demo_input.mp4       # Klip demo pendek (<50MB)
assets/demo_output.mp4      # Output inference
assets/sample_outputs/      # Screenshot anotasi
```

---

## 5. Ringkasan Reproduksibilitas

```bash
# Jalankan evaluasi setelah dataset tersedia di data/mot20/
python src/evaluation/evaluate.py \
    --dataset data/mot20/train/MOT20-01 \
    --conf 0.4 \
    --device cpu
```

> **Catatan:** Kepatuhan terhadap lisensi masing-masing dataset sepenuhnya menjadi tanggung jawab pengguna. Repository ini hanya menyediakan kode eksekusi dan instruksi unduhan.
