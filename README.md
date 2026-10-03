# MobileNetV3-Small Transfer Learning untuk Klasifikasi Objek pada Area Pick and Place Robot UR3

**Fairus - 4222401043**  
**Cluster 4 - Guide Arm Robot**

## 👋 Perkenalan

Repository ini merupakan hasil tugas individu Transfer Learning pada mata kuliah Computer Vision and Deep Learning.

Model utama yang dikerjakan adalah **MobileNetV3-Small** untuk melakukan klasifikasi objek pada area kerja robot UR3 sebagai tahap awal sistem vision pada proses pick and place.

Selain melakukan eksperimen pada MobileNetV3-Small, hasil terbaik model kemudian dibandingkan secara deskriptif dengan **ResNet18** dan **ResNet50** yang dikerjakan oleh anggota project lainnya.

---

## 📌 Deskripsi

Eksperimen MobileNetV3-Small dilakukan menggunakan tiga mode pelatihan:

- **Feature Extraction**
- **Fine-Tuning Partial**
- **Training from Scratch**

Tujuannya adalah mengetahui pengaruh penggunaan pretrained ImageNet dan jumlah layer yang dilatih terhadap performa klasifikasi.

Hasil MobileNetV3-Small kemudian dibandingkan dengan hasil ResNet18 dan ResNet50 untuk melihat perbedaan akurasi, ukuran model, waktu training, dan latency inference.

---

## 📊 Dataset

Dataset MobileNetV3-Small terdiri dari **600 citra** dengan 5 kelas.

| Kelas | Jumlah |
|---|---:|
| `all_blocks` | 120 |
| `balok_hijautua` | 120 |
| `balok_hitam` | 120 |
| `balok_merahmuda` | 120 |
| `papan_balok` | 120 |
| **Total** | **600** |

Pembagian dataset:

| Split | Jumlah |
|---|---:|
| Training | 420 |
| Validation | 90 |
| Testing | 90 |

---

## 🧠 Metodologi MobileNetV3-Small

Konfigurasi umum:

- **Model:** MobileNetV3-Small
- **Input:** 224 × 224 pixel
- **Batch Size:** 16
- **Epoch:** 10
- **Optimizer:** Adam
- **Scheduler:** CosineAnnealingLR
- **Pretrained:** ImageNet
- **Device:** CPU
- **Random Seed:** 42

Normalisasi menggunakan ImageNet:

- Mean = `[0.485, 0.456, 0.406]`
- Std = `[0.229, 0.224, 0.225]`

### Detail 3 Mode

| Mode | Bobot Awal | Layer yang Dilatih | Learning Rate |
|---|---|---|---:|
| Feature Extraction | ImageNet | Classifier | 1e-3 |
| Fine-Tuning Partial | ImageNet | 3 blok feature terakhir + classifier | 1e-4 / 1e-3 |
| Scratch | Random | Seluruh model | 1e-3 |

---

# 📈 Hasil MobileNetV3-Small

## Hasil Training 3 Mode

| Mode | Best Val Acc | Val Loss | Best Epoch | Epoch ≥90% | Training Time | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| Feature Extraction | 96.67% | 0.1274 | 9 | 1 | 58.93 s | 96.67% |
| Fine-Tuning Partial | **100.00%** | **0.0452** | 10 | 2 | **50.91 s** | **97.78%** |
| Scratch | 20.00% | 1.6155 | 2 | Tidak tercapai | 91.73 s | 20.00% |

## Hasil Evaluasi

| Mode | Precision | Recall | F1-Score | Mean Latency | Approx FPS |
|---|---:|---:|---:|---:|---:|
| Feature Extraction | 96.95% | 96.67% | 96.66% | 8.089 ms | 123.63 |
| Fine-Tuning Partial | **98.00%** | **97.78%** | **97.77%** | **7.598 ms** | **131.62** |
| Scratch | 4.00% | 20.00% | 6.67% | 7.807 ms | 128.10 |

> Nilai FPS merupakan estimasi berdasarkan waktu forward inference model dan belum mencakup proses pengambilan frame kamera, preprocessing, komunikasi robot, maupun pergerakan robot.

---

# 📉 Grafik Training

## Feature Extraction - Accuracy

![Feature Accuracy](results/plots/accuracy_mobilenetv3_small_feature.png)

## Feature Extraction - Loss

![Feature Loss](results/plots/loss_mobilenetv3_small_feature.png)

## Fine-Tuning Partial - Accuracy

![Partial Accuracy](results/plots/accuracy_mobilenetv3_small_partial.png)

## Fine-Tuning Partial - Loss

![Partial Loss](results/plots/loss_mobilenetv3_small_partial.png)

## Training from Scratch - Accuracy

![Scratch Accuracy](results/plots/accuracy_mobilenetv3_small_scratch.png)

## Training from Scratch - Loss

![Scratch Loss](results/plots/loss_mobilenetv3_small_scratch.png)

## Perbandingan Validation Accuracy 3 Mode

![Comparison Accuracy](results/plots/comparison_accuracy_3_modes_mobilenetv3_small.png)

## Perbandingan Validation Loss 3 Mode

![Comparison Loss](results/plots/comparison_loss_3_modes_mobilenetv3_small.png)

---

# 🔬 Confusion Matrix

## Feature Extraction

![Feature Confusion Matrix](results/evaluation/confusion_matrix_mobilenetv3_small_feature.png)

## Fine-Tuning Partial

![Partial Confusion Matrix](results/evaluation/confusion_matrix_mobilenetv3_small_partial.png)

## Training from Scratch

![Scratch Confusion Matrix](results/evaluation/confusion_matrix_mobilenetv3_small_scratch.png)

---

# 🏆 Mode MobileNetV3-Small Terbaik

Mode terbaik pada eksperimen ini adalah **Fine-Tuning Partial**.

| Parameter | Hasil |
|---|---:|
| Best Validation Accuracy | **100.00%** |
| Best Validation Loss | **0.0452** |
| Best Epoch | **10** |
| Epoch Pertama ≥90% | **2** |
| Training Time | **50.91 s** |
| Test Accuracy | **97.78%** |
| Precision | **98.00%** |
| Recall | **97.78%** |
| F1-Score | **97.77%** |
| Mean Latency | **7.598 ms/image** |
| Median Latency | **7.610 ms/image** |
| Approx FPS | **131.62 FPS** |

Fine-Tuning Partial memberikan performa terbaik karena tetap memanfaatkan pretrained ImageNet tetapi mengizinkan tiga blok feature terakhir menyesuaikan fitur tingkat tinggi dengan dataset target.

---

# 🔎 Analisis 3 Mode

### Feature Extraction

Feature Extraction menghasilkan test accuracy **96.67%** walaupun hanya classifier yang dilatih.

Hasil ini menunjukkan bahwa fitur dari pretrained ImageNet sudah cukup relevan untuk melakukan klasifikasi objek pada dataset target.

### Fine-Tuning Partial

Fine-Tuning Partial menghasilkan performa terbaik dengan test accuracy **97.78%** dan F1-score **97.77%**.

Dengan membuka tiga blok feature terakhir, model dapat mempertahankan fitur umum dari ImageNet sekaligus menyesuaikan fitur tingkat tinggi terhadap karakteristik objek pada area kerja UR3.

### Training from Scratch

Training from Scratch hanya memperoleh validation dan test accuracy sebesar **20%**.

Training accuracy sempat meningkat lebih dari 80%, tetapi performa validation tetap berada pada 20%. Dengan lima kelas seimbang, nilai tersebut setara dengan peluang memilih satu kelas dari lima kelas.

Hal ini menunjukkan bahwa jumlah data dan 10 epoch yang digunakan belum cukup untuk melatih MobileNetV3-Small secara efektif dari bobot acak.

---

# ⚖️ Perbandingan dengan ResNet18 dan ResNet50

Selain MobileNetV3-Small, project juga menggunakan ResNet18 dan ResNet50 yang dikerjakan oleh anggota lain.

Data yang tersedia menunjukkan hasil berikut:

| Model | Test Accuracy | Approx Parameter | Latency CPU | Training Time |
|---|---:|---:|---:|---:|
| ResNet18 | **100.00%** | ±11.2 juta | 31.4 ms/image | 485 s |
| ResNet50 | **100.00%** | ±23.5 juta | 233.76 ms/image* | 212.43 s* |
| MobileNetV3-Small Partial | 97.78% | **1.52 juta** | **7.598 ms/image** | **50.91 s** |

\* Nilai ResNet50 berasal dari hasil Feature Extraction yang tersedia dari anggota project.

> **Catatan:** Perbandingan ini bersifat deskriptif karena konfigurasi evaluasi antaranggota tidak seluruhnya identik. MobileNetV3-Small dan hasil ResNet18 yang tersedia menggunakan 90 citra test, sedangkan salah satu hasil evaluasi ResNet50 menggunakan 11 citra test. Oleh karena itu, angka akurasi tidak digunakan sebagai perbandingan mutlak antararsitektur.

---

## Analisis Perbandingan Model

### ResNet18

ResNet18 memperoleh test accuracy **100%** pada data pengujian yang tersedia.

Model ini memiliki sekitar **11.2 juta parameter** dan latency CPU sekitar **31.4 ms/image**.

Dibandingkan MobileNetV3-Small, ResNet18 memberikan akurasi test yang lebih tinggi pada hasil eksperimen yang tersedia, tetapi memiliki jumlah parameter dan latency yang lebih besar.

### ResNet50

ResNet50 juga memperoleh test accuracy **100%** pada hasil yang diberikan oleh anggota project.

ResNet50 memiliki sekitar **23.5 juta parameter**, sehingga merupakan arsitektur paling besar dari ketiga model yang dibandingkan.

Pada pengujian Feature Extraction yang tersedia, average latency ResNet50 tercatat sekitar **233.76 ms/image** dengan estimasi sekitar **4.28 FPS**.

Namun, hasil ResNet50 menggunakan konfigurasi data pengujian yang berbeda sehingga perbandingan akurasinya dengan MobileNetV3-Small dan ResNet18 harus diinterpretasikan dengan hati-hati.

### MobileNetV3-Small

MobileNetV3-Small Partial memperoleh test accuracy **97.78%** dengan hanya sekitar **1.52 juta parameter**.

Latency yang diperoleh adalah **7.598 ms/image**, jauh lebih rendah dibandingkan angka latency ResNet18 dan ResNet50 yang tersedia.

Hal ini menunjukkan keunggulan utama MobileNetV3-Small pada efisiensi komputasi.

---

## Ringkasan Perbandingan

Dari hasil yang tersedia:

- **ResNet18** memberikan test accuracy tertinggi pada test set 90 gambar, tetapi memiliki model yang lebih besar dan inference lebih lambat dibandingkan MobileNetV3-Small.
- **ResNet50** juga menghasilkan accuracy tinggi, tetapi memiliki jumlah parameter paling besar dan latency yang lebih tinggi pada pengujian yang tersedia.
- **MobileNetV3-Small** memiliki sedikit penurunan accuracy dibandingkan hasil ResNet18, tetapi memiliki jumlah parameter jauh lebih kecil dan inference yang jauh lebih cepat.

Untuk aplikasi robot yang memiliki keterbatasan sumber daya komputasi dan membutuhkan respons cepat, MobileNetV3-Small memberikan trade-off yang menarik antara accuracy dan efisiensi.

---

# 📦 Struktur Repository

```text
.
├── dataset_raw/
│   ├── all_blocks/
│   ├── balok_hijautua/
│   ├── balok_hitam/
│   ├── balok_merahmuda/
│   └── papan_balok/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── best_mobilenetv3_small_feature.pth
│   ├── best_mobilenetv3_small_partial.pth
│   └── best_mobilenetv3_small_scratch.pth
│
├── results/
│   │
│   ├── history/
│   │   ├── history_mobilenetv3_small_feature.csv
│   │   ├── history_mobilenetv3_small_partial.csv
│   │   └── history_mobilenetv3_small_scratch.csv
│   │
│   ├── plots/
│   │   ├── accuracy_mobilenetv3_small_feature.png
│   │   ├── accuracy_mobilenetv3_small_partial.png
│   │   ├── accuracy_mobilenetv3_small_scratch.png
│   │   ├── loss_mobilenetv3_small_feature.png
│   │   ├── loss_mobilenetv3_small_partial.png
│   │   ├── loss_mobilenetv3_small_scratch.png
│   │   ├── comparison_accuracy_3_modes_mobilenetv3_small.png
│   │   └── comparison_loss_3_modes_mobilenetv3_small.png
│   │
│   └── evaluation/
│       ├── hasil_3_mode_mobilenetv3_small.csv
│       ├── confusion_matrix_mobilenetv3_small_feature.png
│       ├── confusion_matrix_mobilenetv3_small_partial.png
│       └── confusion_matrix_mobilenetv3_small_scratch.png
│
├── train_mobilenet.py
├── train_mobilenet_3_modes.py
├── evaluate_mobilenet.py
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

---

# ✅ Kesimpulan

Eksperimen pada MobileNetV3-Small menunjukkan bahwa **Fine-Tuning Partial** merupakan mode terbaik dengan test accuracy **97.78%**, F1-score **97.77%**, dan mean latency **7.598 ms/image**.

Feature Extraction juga memberikan performa tinggi dengan test accuracy **96.67%**, sedangkan Training from Scratch hanya mencapai **20%**, sehingga penggunaan pretrained ImageNet memberikan keuntungan besar pada dataset yang relatif kecil.

Pada perbandingan dengan hasil ResNet18 dan ResNet50 yang tersedia, MobileNetV3-Small memiliki accuracy sedikit lebih rendah, tetapi unggul dari sisi jumlah parameter dan kecepatan inference.

Dengan karakteristik tersebut, MobileNetV3-Small menjadi alternatif yang efisien untuk pengembangan sistem vision pada robot UR3 yang membutuhkan proses klasifikasi dengan kebutuhan komputasi relatif rendah.