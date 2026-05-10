"""
train.py

Trains both stages sequentially.
Stage 1: gunshot / siren / background
Stage 2: police / ambulance / firetruck / unknown

Saves:
    checkpoints/stage1_best.pth
    checkpoints/stage2_best.pth
    checkpoints/stage1_curves.png
    checkpoints/stage2_curves.png

Run:
    docker compose run app python src/train.py
    docker compose run app python src/train.py --stage 1
    docker compose run app python src/train.py --stage 2
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from model import get_stage1_model, get_stage2_model
from dataset import get_dataloaders


def compute_class_weights(manifest_path, num_classes, device):
    df = pd.read_csv(manifest_path)
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for mel, labels in tqdm(loader, desc="train", leave=False):
        mel, labels = mel.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(mel)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += mel.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for mel, labels in loader:
            mel, labels = mel.to(device), labels.to(device)
            logits = model(mel)
            loss = criterion(logits, labels)
            total_loss += loss.item() * mel.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += mel.size(0)
    return total_loss / total, correct / total


def save_curves(train_losses, val_losses, train_accs, val_accs, name):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="train")
    axes[0].plot(val_losses,   label="val")
    axes[0].set_title(f"{name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(train_accs, label="train")
    axes[1].plot(val_accs,   label="val")
    axes[1].set_title(f"{name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"{name}_curves.png"))
    plt.close()


def train_stage(stage_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Training Stage {stage_num}")
    print(f"{'='*50}")
    print(f"Device: {device}")

    if stage_num == 1:
        manifest_path = os.path.join(config.PROCESSED_STAGE1_DIR, "manifest.csv")
        model         = get_stage1_model().to(device)
        num_classes   = config.NUM_STAGE1_CLASSES
        class_names   = config.STAGE1_CLASSES
        ckpt_name     = "stage1_best.pth"
        curve_name    = "stage1"
    else:
        manifest_path = os.path.join(config.PROCESSED_STAGE2_DIR, "manifest.csv")
        model         = get_stage2_model().to(device)
        num_classes   = config.NUM_STAGE2_CLASSES
        class_names   = config.STAGE2_CLASSES
        ckpt_name     = "stage2_best.pth"
        curve_name    = "stage2"

    train_loader, val_loader, test_loader = get_dataloaders(manifest_path)

    weights   = compute_class_weights(manifest_path, num_classes, device)
    print("Class weights:")
    for cls, w in zip(class_names, weights):
        print(f"  {cls:<14}: {w:.4f}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:03d}/{config.EPOCHS} | "
            f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> checkpoint saved (val_acc={val_acc:.4f})")

    save_curves(train_losses, val_losses, train_accs, val_accs, curve_name)
    print(f"\nStage {stage_num} done. Best val accuracy: {best_val_acc:.4f}")

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=0,
                        help="Train only stage 1 or 2. Omit to train both.")
    args = parser.parse_args()

    if args.stage == 1:
        train_stage(1)
    elif args.stage == 2:
        train_stage(2)
    else:
        train_stage(1)
        train_stage(2)