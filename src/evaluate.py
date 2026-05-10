"""
evaluate.py

Evaluates both stages independently and then runs the full two-stage
pipeline on the Stage 1 test set to show end-to-end performance.

Run:
    docker compose run app python src/evaluate.py
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import config
from model import get_stage1_model, get_stage2_model
from dataset import get_dataloaders, AudioDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from dataset import _get_groups


def load_model(model, ckpt_name, device):
    path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def evaluate_stage(stage_num, device):
    print(f"\n{'='*50}")
    print(f"Stage {stage_num} Evaluation")
    print(f"{'='*50}")

    if stage_num == 1:
        manifest_path = os.path.join(config.PROCESSED_STAGE1_DIR, "manifest.csv")
        model         = load_model(get_stage1_model(), "stage1_best.pth", device)
        class_names   = config.STAGE1_CLASSES
        cm_name       = "stage1_confusion_matrix.png"
    else:
        manifest_path = os.path.join(config.PROCESSED_STAGE2_DIR, "manifest.csv")
        model         = load_model(get_stage2_model(), "stage2_best.pth", device)
        class_names   = config.STAGE2_CLASSES
        cm_name       = "stage2_confusion_matrix.png"

    _, _, test_loader = get_dataloaders(manifest_path)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(device)
            logits = model(mel)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Stage {stage_num} Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(config.CHECKPOINT_DIR, cm_name)
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


def evaluate_pipeline(device):
    """
    Runs the full two-stage pipeline:
    - Stage 1 predicts gunshot / siren / background
    - For siren predictions, Stage 2 predicts the siren type
    - If Stage 2 confidence < threshold, returns 'unknown'
    """
    print(f"\n{'='*50}")
    print("Full Pipeline Evaluation")
    print(f"{'='*50}")

    stage1 = load_model(get_stage1_model(), "stage1_best.pth", device)
    stage2 = load_model(get_stage2_model(), "stage2_best.pth", device)

    manifest_path = os.path.join(config.PROCESSED_STAGE1_DIR, "manifest.csv")
    _, _, test_loader = get_dataloaders(manifest_path)

    results = []
    siren_idx = config.STAGE1_CLASSES.index("siren")

    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(device)
            s1_logits = stage1(mel)
            s1_probs  = F.softmax(s1_logits, dim=1)
            s1_preds  = s1_probs.argmax(1)

            for i in range(mel.size(0)):
                true_label  = labels[i].item()
                s1_pred     = s1_preds[i].item()
                s1_class    = config.STAGE1_CLASSES[s1_pred]

                if s1_pred == siren_idx:
                    s2_logits = stage2(mel[i].unsqueeze(0))
                    s2_probs  = F.softmax(s2_logits, dim=1)
                    s2_conf   = s2_probs.max().item()
                    s2_pred   = s2_probs.argmax(1).item()
                    s2_class  = config.STAGE2_CLASSES[s2_pred] if s2_conf >= config.STAGE2_CONFIDENCE_THRESHOLD else "unknown"
                    final     = f"siren/{s2_class}"
                else:
                    final = s1_class

                results.append({
                    "true_stage1":  config.STAGE1_CLASSES[true_label],
                    "s1_predicted": s1_class,
                    "final":        final,
                })

    total  = len(results)
    s1_correct = sum(1 for r in results if r["true_stage1"] == r["s1_predicted"])
    print(f"Stage 1 accuracy on test set : {s1_correct/total:.4f} ({s1_correct}/{total})")

    siren_results = [r for r in results if r["true_stage1"] == "siren"]
    if siren_results:
        typed   = sum(1 for r in siren_results if "unknown" not in r["final"])
        unknown = sum(1 for r in siren_results if "unknown" in r["final"])
        print(f"\nOf {len(siren_results)} true siren samples routed to Stage 2:")
        print(f"  Typed confidently : {typed}")
        print(f"  Returned unknown  : {unknown}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_stage(1, device)
    evaluate_stage(2, device)
    evaluate_pipeline(device)