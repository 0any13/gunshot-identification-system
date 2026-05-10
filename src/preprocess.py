"""
preprocess.py

Computes mel spectrograms for both stages and saves manifests.

Run:
    docker compose run app python src/preprocess.py
"""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

import config


def load_and_pad(path):
    target_len = config.SAMPLE_RATE * config.CLIP_DURATION
    audio, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    if len(audio) >= target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    return audio


def compute_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        fmax=config.F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def process_stage(stage_name, class_list, raw_dir, processed_dir):
    print(f"\n--- {stage_name} ---")
    os.makedirs(processed_dir, exist_ok=True)

    label_map = {cls: idx for idx, cls in enumerate(class_list)}
    records = []

    for class_name in class_list:
        class_dir = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  [WARNING] not found: {class_dir}")
            continue

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))
        ]
        print(f"  Processing {class_name}: {len(files)} files")

        for fname in tqdm(files, desc=class_name):
            src_path = os.path.join(class_dir, fname)
            try:
                audio = load_and_pad(src_path)
                mel   = compute_mel(audio)

                stem     = os.path.splitext(fname)[0]
                out_name = f"{class_name}_{stem}.npy"
                out_path = os.path.join(processed_dir, out_name)
                np.save(out_path, mel)

                records.append({
                    "npy_path":    out_path,
                    "label":       label_map[class_name],
                    "class_name":  class_name,
                    "source_file": fname,
                })
            except Exception as e:
                print(f"  [ERROR] {src_path}: {e}")

    manifest = pd.DataFrame(records)
    manifest_path = os.path.join(processed_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\n  {stage_name} done. {len(manifest)} samples.")
    print(manifest["class_name"].value_counts().to_string())
    return manifest_path


if __name__ == "__main__":
    process_stage(
        stage_name    = "Stage 1 (coarse)",
        class_list    = config.STAGE1_CLASSES,
        raw_dir       = "data/raw/stage1",
        processed_dir = config.PROCESSED_STAGE1_DIR,
    )
    process_stage(
        stage_name    = "Stage 2 (siren types)",
        class_list    = config.STAGE2_CLASSES,
        raw_dir       = "data/raw/stage2",
        processed_dir = config.PROCESSED_STAGE2_DIR,
    )