"""
train.py

Training loop with:
- cross-entropy loss
- Adam optimizer with learning rate scheduler
- best-checkpoint saving based on validation accuracy
- loss/accuracy curves saved to checkpoints/

Run:
    python src/train.py
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from model import AudioCNN
from dataset import get_dataloaders


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
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
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
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += mel.size(0)

    return total_loss / total, correct / total


def save_curves(train_losses, val_losses, train_accs, val_accs):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="train")
    axes[0].plot(val_losses, label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(train_accs, label="train")
    axes[1].plot(val_accs, label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, "training_curves.png"))
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifest_path = os.path.join(config.PROCESSED_DATA_DIR, "manifest.csv")
    train_loader, val_loader, test_loader = get_dataloaders(manifest_path)

    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # reduce LR by 0.5 if val loss does not improve for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
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
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best checkpoint (val_acc={val_acc:.4f})")

    save_curves(train_losses, val_losses, train_accs, val_accs)
    print(f"\nTraining done. Best val accuracy: {best_val_acc:.4f}")

    # evaluate on held-out test set
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pth")))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
