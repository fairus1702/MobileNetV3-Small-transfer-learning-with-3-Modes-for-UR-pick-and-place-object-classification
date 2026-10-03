import os
import time
import copy
import random

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
    ConfusionMatrixDisplay
)


# ============================================================
# KONFIGURASI
# ============================================================

SEED = 42

BATCH_SIZE = 16
EPOCHS = 10

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

MODEL_DIR = "models"
HISTORY_DIR = "results/history"
PLOT_DIR = "results/plots"
EVALUATION_DIR = "results/evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

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
# PREPROCESSING
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


eval_transform = transforms.Compose([
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
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# INFO DATASET
# ============================================================

print("=" * 75)
print("MOBILENETV3-SMALL - 3 MODE EXPERIMEN")
print("=" * 75)

print(f"Device          : {DEVICE}")
print(f"Classes         : {CLASS_NAMES}")
print(f"Jumlah kelas    : {NUM_CLASSES}")
print(f"Training images : {len(train_dataset)}")
print(f"Validation      : {len(val_dataset)}")
print(f"Testing         : {len(test_dataset)}")
print(f"Epoch           : {EPOCHS}")
print(f"Batch Size      : {BATCH_SIZE}")

print("=" * 75)


# ============================================================
# MEMBUAT MODEL
# ============================================================

def create_model(mode):

    # --------------------------------------------------------
    # MODE 1: FEATURE EXTRACTION
    # --------------------------------------------------------

    if mode == "feature":

        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # Freeze seluruh model
        for param in model.parameters():
            param.requires_grad = False

        in_features = model.classifier[3].in_features

        model.classifier[3] = nn.Linear(
            in_features,
            NUM_CLASSES
        )

        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=1e-3
        )


    # --------------------------------------------------------
    # MODE 2: PARTIAL FINE-TUNING
    # --------------------------------------------------------

    elif mode == "partial":

        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # Freeze semuanya
        for param in model.parameters():
            param.requires_grad = False

        # Buka 3 blok feature terakhir
        for block in model.features[-3:]:

            for param in block.parameters():
                param.requires_grad = True

        in_features = model.classifier[3].in_features

        model.classifier[3] = nn.Linear(
            in_features,
            NUM_CLASSES
        )

        optimizer = torch.optim.Adam([
            {
                "params": model.features[-3:].parameters(),
                "lr": 1e-4
            },
            {
                "params": model.classifier.parameters(),
                "lr": 1e-3
            }
        ])


    # --------------------------------------------------------
    # MODE 3: SCRATCH
    # --------------------------------------------------------

    elif mode == "scratch":

        model = models.mobilenet_v3_small(
            weights=None
        )

        in_features = model.classifier[3].in_features

        model.classifier[3] = nn.Linear(
            in_features,
            NUM_CLASSES
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )


    else:

        raise ValueError(
            "Mode harus feature, partial, atau scratch."
        )


    model = model.to(DEVICE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    return model, optimizer, scheduler


# ============================================================
# PARAMETER MODEL
# ============================================================

def get_parameter_count(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# BATCHNORM FROZEN
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
                    p.requires_grad
                    for p in parameters
                )

                if not trainable:
                    module.eval()


# ============================================================
# TRAIN 1 EPOCH
# ============================================================

def train_one_epoch(
    model,
    criterion,
    optimizer
):

    model.train()

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


    loss = running_loss / total

    accuracy = (
        correct / total
    ) * 100


    return loss, accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    criterion
):

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


    loss = running_loss / total

    accuracy = (
        correct / total
    ) * 100


    return loss, accuracy


# ============================================================
# TEST
# ============================================================

def test_model(model):

    model.eval()

    labels_all = []
    predictions_all = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            labels_all.extend(
                labels.numpy()
            )

            predictions_all.extend(
                predicted.cpu().numpy()
            )


    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    precision = precision_score(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0
    )


    return (
        labels_all,
        predictions_all,
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100
    )


# ============================================================
# LATENCY
# ============================================================

def measure_latency(model):

    model.eval()

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


    # Warm-up
    with torch.no_grad():

        for _ in range(10):
            _ = model(sample_image)


    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


    latencies = []


    with torch.no_grad():

        for image, _ in latency_loader:

            image = image.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(image)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )


    mean_latency = np.mean(
        latencies
    )

    median_latency = np.median(
        latencies
    )

    fps = (
        1000 / mean_latency
        if mean_latency > 0
        else 0
    )


    return (
        mean_latency,
        median_latency,
        fps
    )


# ============================================================
# TRAIN SATU MODE
# ============================================================

def run_mode(mode):

    print()
    print("=" * 75)
    print(
        f"MODE: {mode.upper()}"
    )
    print("=" * 75)


    # Reset seed agar pembandingan lebih konsisten
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    model, optimizer, scheduler = create_model(
        mode
    )


    total_params, trainable_params = (
        get_parameter_count(model)
    )


    print(
        f"Total Parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable Parameters : "
        f"{trainable_params:,}"
    )


    criterion = nn.CrossEntropyLoss()


    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time_seconds": []
    }


    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    best_epoch = 0

    epoch_reach_90 = None

    best_weights = copy.deepcopy(
        model.state_dict()
    )


    training_start = time.perf_counter()


    # ========================================================
    # LOOP TRAINING
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.perf_counter()


        train_loss, train_acc = train_one_epoch(
            model,
            criterion,
            optimizer
        )


        val_loss, val_acc = validate(
            model,
            criterion
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
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_acc
        )

        history["epoch_time_seconds"].append(
            epoch_time
        )


        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )


        # Epoch pertama validation accuracy >= 90%
        if (
            epoch_reach_90 is None
            and val_acc >= 90
        ):
            epoch_reach_90 = epoch


        # Pilih model berdasarkan:
        # 1. validation accuracy tertinggi
        # 2. jika sama, validation loss terendah
        if (
            val_acc > best_val_accuracy
            or (
                val_acc == best_val_accuracy
                and val_loss < best_val_loss
            )
        ):

            best_val_accuracy = val_acc
            best_val_loss = val_loss
            best_epoch = epoch

            best_weights = copy.deepcopy(
                model.state_dict()
            )


    total_training_time = (
        time.perf_counter()
        - training_start
    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_weights
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        f"best_mobilenetv3_small_{mode}.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    history_path = os.path.join(
        HISTORY_DIR,
        f"history_mobilenetv3_small_{mode}.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )


    # ========================================================
    # TEST BEST MODEL
    # ========================================================

    (
        labels_all,
        predictions_all,
        test_accuracy,
        precision,
        recall,
        f1
    ) = test_model(model)


    # ========================================================
    # LATENCY
    # ========================================================

    (
        mean_latency,
        median_latency,
        fps
    ) = measure_latency(model)


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        labels_all,
        predictions_all
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
        f"MobileNetV3-Small - {mode.capitalize()}"
    )

    plt.tight_layout()


    confusion_path = os.path.join(
        EVALUATION_DIR,
        f"confusion_matrix_mobilenetv3_small_{mode}.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300
    )

    plt.close()


    # ========================================================
    # PLOT ACCURACY MODE
    # ========================================================

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

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.title(
        f"MobileNetV3-Small - {mode.capitalize()} Accuracy"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()


    accuracy_path = os.path.join(
        PLOT_DIR,
        f"accuracy_mobilenetv3_small_{mode}.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300
    )

    plt.close()


    # ========================================================
    # PLOT LOSS MODE
    # ========================================================

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

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"MobileNetV3-Small - {mode.capitalize()} Loss"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()


    loss_path = os.path.join(
        PLOT_DIR,
        f"loss_mobilenetv3_small_{mode}.png"
    )

    plt.savefig(
        loss_path,
        dpi=300
    )

    plt.close()


    # ========================================================
    # HASIL MODE
    # ========================================================

    print()
    print("-" * 75)

    print(
        f"Best Val Accuracy : "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Best Val Loss     : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best Epoch        : "
        f"{best_epoch}"
    )

    print(
        f"Epoch >=90%       : "
        f"{epoch_reach_90}"
    )

    print(
        f"Training Time     : "
        f"{total_training_time:.2f} detik"
    )

    print(
        f"Test Accuracy     : "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Precision         : "
        f"{precision:.2f}%"
    )

    print(
        f"Recall            : "
        f"{recall:.2f}%"
    )

    print(
        f"F1-Score          : "
        f"{f1:.2f}%"
    )

    print(
        f"Mean Latency      : "
        f"{mean_latency:.3f} ms/image"
    )

    print(
        f"Median Latency    : "
        f"{median_latency:.3f} ms/image"
    )

    print(
        f"Approx FPS        : "
        f"{fps:.2f}"
    )

    print("-" * 75)


    return {

        "mode": mode,

        "best_val_accuracy": (
            best_val_accuracy
        ),

        "best_val_loss": (
            best_val_loss
        ),

        "best_epoch": (
            best_epoch
        ),

        "epoch_first_90": (
            epoch_reach_90
            if epoch_reach_90 is not None
            else "Not reached"
        ),

        "training_time_seconds": (
            total_training_time
        ),

        "test_accuracy": (
            test_accuracy
        ),

        "precision_macro": (
            precision
        ),

        "recall_macro": (
            recall
        ),

        "f1_score_macro": (
            f1
        ),

        "mean_latency_ms": (
            mean_latency
        ),

        "median_latency_ms": (
            median_latency
        ),

        "approx_fps": (
            fps
        ),

        "total_parameters": (
            total_params
        ),

        "trainable_parameters": (
            trainable_params
        )
    }, history


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    MODES = [
        "feature",
        "partial",
        "scratch"
    ]

    all_results = []

    histories = {}


    for mode in MODES:

        result, history = run_mode(
            mode
        )

        all_results.append(
            result
        )

        histories[mode] = history


    # ========================================================
    # SAVE HASIL 3 MODE
    # ========================================================

    result_df = pd.DataFrame(
        all_results
    )


    result_csv = os.path.join(
        EVALUATION_DIR,
        "hasil_3_mode_mobilenetv3_small.csv"
    )


    result_df.to_csv(
        result_csv,
        index=False
    )


    # ========================================================
    # GRAFIK VALIDATION ACCURACY 3 MODE
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )


    for mode in MODES:

        plt.plot(
            histories[mode]["epoch"],
            histories[mode]["val_accuracy"],
            marker="o",
            label=mode.capitalize()
        )


    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title(
        "Perbandingan Validation Accuracy "
        "3 Mode MobileNetV3-Small"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()


    comparison_accuracy_path = os.path.join(
        PLOT_DIR,
        "comparison_accuracy_3_modes_mobilenetv3_small.png"
    )


    plt.savefig(
        comparison_accuracy_path,
        dpi=300
    )

    plt.close()


    # ========================================================
    # GRAFIK VALIDATION LOSS 3 MODE
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )


    for mode in MODES:

        plt.plot(
            histories[mode]["epoch"],
            histories[mode]["val_loss"],
            marker="o",
            label=mode.capitalize()
        )


    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")

    plt.title(
        "Perbandingan Validation Loss "
        "3 Mode MobileNetV3-Small"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()


    comparison_loss_path = os.path.join(
        PLOT_DIR,
        "comparison_loss_3_modes_mobilenetv3_small.png"
    )


    plt.savefig(
        comparison_loss_path,
        dpi=300
    )

    plt.close()


    # ========================================================
    # PILIH MODE TERBAIK
    # ========================================================

    best_result = sorted(
        all_results,
        key=lambda x: (
            -x["best_val_accuracy"],
            x["best_val_loss"]
        )
    )[0]


    print()
    print("=" * 75)
    print("HASIL AKHIR 3 MODE MOBILENETV3-SMALL")
    print("=" * 75)

    print(
        result_df[
            [
                "mode",
                "best_val_accuracy",
                "best_epoch",
                "epoch_first_90",
                "training_time_seconds",
                "test_accuracy",
                "mean_latency_ms",
                "approx_fps"
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print("=" * 75)
    print("MODE TERBAIK")
    print("=" * 75)

    print(
        f"Mode              : "
        f"{best_result['mode']}"
    )

    print(
        f"Best Val Accuracy : "
        f"{best_result['best_val_accuracy']:.2f}%"
    )

    print(
        f"Best Val Loss     : "
        f"{best_result['best_val_loss']:.4f}"
    )

    print(
        f"Test Accuracy     : "
        f"{best_result['test_accuracy']:.2f}%"
    )

    print(
        f"Mean Latency      : "
        f"{best_result['mean_latency_ms']:.3f} ms/image"
    )

    print(
        f"Approx FPS        : "
        f"{best_result['approx_fps']:.2f}"
    )

    print()
    print(
        f"Hasil CSV : "
        f"{result_csv}"
    )

    print(
        f"Grafik Accuracy : "
        f"{comparison_accuracy_path}"
    )

    print(
        f"Grafik Loss     : "
        f"{comparison_loss_path}"
    )

    print("=" * 75)