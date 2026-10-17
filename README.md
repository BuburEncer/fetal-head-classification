# Fetal Head Plane Classification — ResNet50

Klasifikasi plane kepala janin pada citra ultrasound (USG) menggunakan transfer learning **ResNet50** dengan **Grad-CAM** untuk explainable AI.

## Ringkasan

| Metrik | Nilai |
|---|---|
| **Accuracy (test)** | **0.83** |
| Weighted F1 | 0.83 |
| Best Val Accuracy | 0.8525 (epoch 34) |
| Split | 70% train / 15% val / 15% test (seed 42) |
| Total gambar | 2,845 (1986 train) |

## Dataset

3 kelas plane USG kepala janin:

| Kelas | Train | Val | Test |
|---|---|---|---|
| Transcerebellar | 478 | 103 | 103 |
| Transthalamic | 1095 | 235 | 235 |
| Transventricular | 413 | 89 | 89 |

Dataset imbalanced (transthalamic dominan). Ditangani dengan **WeightedRandomSampler** + **smoothed class weights** (sqrt-normalized).

## Metrik per kelas (test set, 427 gambar)

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
| Transcerebellar | 0.86 | 0.91 | **0.89** |
| Transthalamic | 0.88 | 0.81 | **0.84** |
| Transventricular | 0.70 | 0.80 | **0.74** |

Kelas transthalamic paling handal; transventricular paling lemah (precision 0.70) — pola umum karena overlap visual antar plane.

## Model

- **Backbone:** ResNet50 pretrained ImageNet
- **Transfer learning:** freeze semua layer, unfreeze `layer3` + `layer4`
- **Head:** `Dropout(0.4) → FC(256) → ReLU → Dropout(0.3) → FC(3)`
- **Augmentasi (train):** flip horizontal/vertikal, rotasi 20°, color jitter, grayscale → to grayscale 3-channel
- **Loss:** CrossEntropyLoss berbobot kelas
- **Optimizer:** Adam, lr=1e-4, StepLR (step 5, gamma 0.5)
- **Early stopping:** patience 7, max 40 epoch (berhenti di epoch 34)
- **Input:** 224×224

## XAI — Grad-CAM

Grad-CAM memvisualisasikan area citra yang paling memengaruhi keputusan prediksi per kelas, memetakan aktivasi `layer4[-1]` ke heatmap. Implementasi lokal (tanpa dependency eksternal, jalan di lingkungan internet-off seperti Kaggle).

Output: `outputs/gradcam_result.png`

## Evaluasi hasil

- Training accuracy ↗ 0.96, val ~0.84 di akhir — overfit ringan (gap ±0.12) meski ada dropout + augmentasi.
- Test accuracy **0.83**, weighted F1 0.83.
- Cross-val konsisten: v7 test 0.84, v8 test 0.83 → model stabil, bukan artefak satu split.

## Reproduksi

Notebook siap di Kaggle GPU. Requirement sudah di-bundle (Grad-CAM lokal, tidak butuh internet untuk install).

```bash
kaggle kernels push -p .
```

## Struktur file

- `fetal-head-classification.ipynb` — notebook lengkap (train, eval, Grad-CAM)
- `kernel-metadata.json` — konfigurasi push Kaggle
- `raw_data/` — dataset mentah (3 folder kelas)
- `outputs/` — plot hasil ditulis saat run