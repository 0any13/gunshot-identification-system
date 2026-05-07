"""
evaluate.py

Loads the best checkpoint and produces:
- confusion matrix (saved as PNG)
- per-class precision, recall, F1

Run:
    python src/evaluate.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import config
from model import AudioCNN
from dataset import get_dataloaders


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = os.path.join(config.PROCESSED_DATA_DIR, "manifest.csv")
    _, _, test_loader = get_dataloaders(manifest_path)

    model = AudioCNN().to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(device)
            logits = model(mel)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=config.CLASSES))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASSES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    out_path = os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(out_path)
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    run_evaluation()
