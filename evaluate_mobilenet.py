import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# KONFIGURASI
# ============================================================

TEST_DIR = "dataset/test"

MODEL_PATH = "models/best_mobilenetv3_small.pth"

EVALUATION_DIR = "results/evaluation"

os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 16


# ============================================================
# PREPROCESSING TEST
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

CLASS_NAMES = test_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)


print("=" * 70)
print("EVALUATION MOBILENETV3-SMALL")
print("=" * 70)

print(f"Device      : {DEVICE}")
print(f"Test images : {len(test_dataset)}")
print(f"Jumlah kelas: {NUM_CLASSES}")
print(f"Classes     : {CLASS_NAMES}")

print("=" * 70)


# ============================================================
# CEK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = models.mobilenet_v3_small(
    weights=None
)

in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    in_features,
    NUM_CLASSES
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    state_dict
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# TESTING
# ============================================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# ============================================================
# METRIK
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# LATENCY
# ============================================================

latency_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


sample_image, _ = next(
    iter(latency_loader)
)

sample_image = sample_image.to(
    DEVICE
)


# ============================================================
# WARM-UP
# ============================================================

with torch.no_grad():

    for _ in range(10):

        _ = model(
            sample_image
        )


if DEVICE.type == "cuda":
    torch.cuda.synchronize()


# ============================================================
# UKUR LATENCY
# ============================================================

latencies = []


with torch.no_grad():

    for image, _ in latency_loader:

        image = image.to(
            DEVICE
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        _ = model(
            image
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        latencies.append(
            latency_ms
        )


average_latency = np.mean(
    latencies
)

min_latency = np.min(
    latencies
)

max_latency = np.max(
    latencies
)

std_latency = np.std(
    latencies
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    "Confusion Matrix MobileNetV3-Small"
)

plt.tight_layout()


confusion_path = os.path.join(
    EVALUATION_DIR,
    "confusion_matrix_mobilenetv3_small.png"
)

plt.savefig(
    confusion_path,
    dpi=300
)

plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()


report_path = os.path.join(
    EVALUATION_DIR,
    "classification_report_mobilenetv3_small.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# SAVE HASIL UTAMA
# ============================================================

results = pd.DataFrame({
    "model": [
        "MobileNetV3-Small"
    ],

    "test_accuracy": [
        accuracy * 100
    ],

    "precision_macro": [
        precision * 100
    ],

    "recall_macro": [
        recall * 100
    ],

    "f1_score_macro": [
        f1 * 100
    ],

    "latency_mean_ms": [
        average_latency
    ],

    "latency_min_ms": [
        min_latency
    ],

    "latency_max_ms": [
        max_latency
    ],

    "latency_std_ms": [
        std_latency
    ],

    "device": [
        str(DEVICE)
    ],

    "test_images": [
        len(test_dataset)
    ]
})


result_path = os.path.join(
    EVALUATION_DIR,
    "hasil_mobilenetv3_small.csv"
)

results.to_csv(
    result_path,
    index=False
)


# ============================================================
# SAVE PREDICTION DETAIL
# ============================================================

prediction_results = pd.DataFrame({
    "actual_class": [
        CLASS_NAMES[label]
        for label in all_labels
    ],

    "predicted_class": [
        CLASS_NAMES[pred]
        for pred in all_predictions
    ],

    "correct": [
        actual == pred
        for actual, pred
        in zip(
            all_labels,
            all_predictions
        )
    ]
})


prediction_path = os.path.join(
    EVALUATION_DIR,
    "prediction_mobilenetv3_small.csv"
)

prediction_results.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print()

print("=" * 70)
print("HASIL TEST MOBILENETV3-SMALL")
print("=" * 70)

print(
    f"Test Accuracy : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision     : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall        : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1-Score      : "
    f"{f1 * 100:.2f}%"
)

print()

print(
    f"Mean Latency  : "
    f"{average_latency:.3f} ms/image"
)

print(
    f"Min Latency   : "
    f"{min_latency:.3f} ms/image"
)

print(
    f"Max Latency   : "
    f"{max_latency:.3f} ms/image"
)

print(
    f"Std Latency   : "
    f"{std_latency:.3f} ms"
)

print(
    f"Device        : "
    f"{DEVICE}"
)

print()

print("FILE HASIL")
print("-" * 70)

print(
    f"Hasil utama          : "
    f"{result_path}"
)

print(
    f"Classification report: "
    f"{report_path}"
)

print(
    f"Prediction detail     : "
    f"{prediction_path}"
)

print(
    f"Confusion matrix      : "
    f"{confusion_path}"
)

print("=" * 70)