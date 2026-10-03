import os
import time
import random
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# KONFIGURASI
# ============================================================

SEED = 42

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

MODEL_DIR = "models"
HISTORY_DIR = "results/history"
PLOT_DIR = "results/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PREPROCESSING & AUGMENTATION
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

CLASS_NAMES = train_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# INFORMASI DATASET
# ============================================================

print("=" * 70)
print("TRAINING MOBILENETV3-SMALL - TRANSFER LEARNING")
print("=" * 70)

print(f"Device          : {DEVICE}")
print(f"Jumlah kelas    : {NUM_CLASSES}")
print(f"Nama kelas      : {CLASS_NAMES}")
print(f"Training images : {len(train_dataset)}")
print(f"Validation      : {len(val_dataset)}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epoch           : {EPOCHS}")
print(f"Learning rate   : {LEARNING_RATE}")

print("=" * 70)


# ============================================================
# LOAD MOBILENETV3-SMALL PRETRAINED
# ============================================================

weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1

model = models.mobilenet_v3_small(
    weights=weights
)


# ============================================================
# FREEZE BACKBONE
# ============================================================

for param in model.parameters():
    param.requires_grad = False


# ============================================================
# GANTI CLASSIFIER
# ============================================================

in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# CEK PARAMETER
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print("MODEL INFORMATION")
print("-" * 70)

print(f"Model              : MobileNetV3-Small")
print(f"Pretrained          : ImageNet")
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print("-" * 70)


# ============================================================
# LOSS & OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# FROZEN BATCHNORM
# ============================================================

def set_frozen_batchnorm_eval(model):

    for module in model.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ):

            parameters = list(
                module.parameters()
            )

            if len(parameters) > 0:

                trainable = any(
                    param.requires_grad
                    for param in parameters
                )

                if not trainable:
                    module.eval()


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    # BatchNorm backbone frozen tetap eval
    set_frozen_batchnorm_eval(model)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(
            outputs,
            1
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        100 * correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            correct += (
                predicted == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        100 * correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# HISTORY
# ============================================================

history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "epoch_time_seconds": []
}


# ============================================================
# BEST MODEL
# ============================================================

best_val_accuracy = 0.0
best_val_loss = float("inf")
best_epoch = 0

epoch_reach_90 = None

best_model_weights = copy.deepcopy(
    model.state_dict()
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("MULAI TRAINING")
print("=" * 70)

training_start = time.perf_counter()


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.perf_counter()

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy = (
        validate()
    )

    scheduler.step()

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )


    history["epoch"].append(
        epoch
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["epoch_time_seconds"].append(
        epoch_time
    )


    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )


    # ================================================
    # CATAT EPOCH PERTAMA >= 90%
    # ================================================

    if (
        epoch_reach_90 is None
        and val_accuracy >= 90
    ):
        epoch_reach_90 = epoch


    # ================================================
    # SIMPAN MODEL TERBAIK
    # ================================================

    if (
        val_accuracy > best_val_accuracy
        or (
            val_accuracy == best_val_accuracy
            and val_loss < best_val_loss
        )
    ):

        best_val_accuracy = val_accuracy
        best_val_loss = val_loss
        best_epoch = epoch

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


# ============================================================
# TOTAL TRAINING TIME
# ============================================================

total_training_time = (
    time.perf_counter()
    - training_start
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "best_mobilenetv3_small.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


# ============================================================
# SAVE HISTORY CSV
# ============================================================

history_df = pd.DataFrame(
    history
)

history_path = os.path.join(
    HISTORY_DIR,
    "history_mobilenetv3_small.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "model": [
        "MobileNetV3-Small"
    ],

    "pretrained": [
        "ImageNet"
    ],

    "strategy": [
        "Feature Extraction"
    ],

    "best_val_accuracy": [
        best_val_accuracy
    ],

    "best_epoch": [
        best_epoch
    ],

    "epoch_reach_90": [
        (
            epoch_reach_90
            if epoch_reach_90
            is not None
            else "Not reached"
        )
    ],

    "training_time_seconds": [
        total_training_time
    ],

    "train_images": [
        len(train_dataset)
    ],

    "val_images": [
        len(val_dataset)
    ],

    "num_classes": [
        NUM_CLASSES
    ]
}

summary_df = pd.DataFrame(
    summary
)

summary_path = os.path.join(
    HISTORY_DIR,
    "summary_mobilenetv3_small.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# GRAFIK ACCURACY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["train_accuracy"],
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    history["epoch"],
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "MobileNetV3-Small Accuracy per Epoch"
)

plt.legend()
plt.grid()

plt.tight_layout()

accuracy_path = os.path.join(
    PLOT_DIR,
    "accuracy_mobilenetv3_small.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


# ============================================================
# GRAFIK LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "MobileNetV3-Small Loss per Epoch"
)

plt.legend()
plt.grid()

plt.tight_layout()

loss_path = os.path.join(
    PLOT_DIR,
    "loss_mobilenetv3_small.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.close()


# ============================================================
# HASIL AKHIR
# ============================================================

print()
print("=" * 70)
print("HASIL TRAINING MOBILENETV3-SMALL")
print("=" * 70)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Epoch               : "
    f"{best_epoch}"
)

if epoch_reach_90 is None:

    print(
        "Epoch mencapai >= 90%    : "
        "Tidak tercapai"
    )

else:

    print(
        f"Epoch mencapai >= 90%    : "
        f"{epoch_reach_90}"
    )

print(
    f"Total Training Time       : "
    f"{total_training_time:.2f} detik"
)

print()
print(
    f"Model tersimpan   : "
    f"{model_path}"
)

print(
    f"History tersimpan : "
    f"{history_path}"
)

print(
    f"Summary tersimpan : "
    f"{summary_path}"
)

print(
    f"Grafik accuracy   : "
    f"{accuracy_path}"
)

print(
    f"Grafik loss       : "
    f"{loss_path}"
)

print("=" * 70)