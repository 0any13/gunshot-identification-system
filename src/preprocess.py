"""
preprocess.py

Reads raw audio files, computes mel spectrograms, and saves them as .npy files
alongside a CSV manifest so the Dataset class can load them efficiently.

Expected raw data layout:
    data/raw/
        gunshot/        <- .wav files
        police_siren/
        ambulance_siren/
        firetruck_siren/
        background/

Run:
    python src/preprocess.py
"""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

import config


def load_and_pad(path, sr=config.SAMPLE_RATE, duration=config.CLIP_DURATION):
    target_len = sr * duration
    audio, _ = librosa.load(path, sr=sr, mono=True)

    if len(audio) >= target_len:
        audio = audio[:target_len]
    else:
        pad = target_len - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")

    return audio


def compute_mel(audio, sr=config.SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        fmax=config.F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def process_dataset():
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    records = []
    label_map = {cls: idx for idx, cls in enumerate(config.CLASSES)}

    for class_name in config.CLASSES:
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARNING] directory not found, skipping: {class_dir}")
            continue

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))
        ]

        print(f"Processing {class_name}: {len(files)} files")

        for fname in tqdm(files, desc=class_name):
            src_path = os.path.join(class_dir, fname)
            try:
                audio = load_and_pad(src_path)
                mel = compute_mel(audio)

                stem = os.path.splitext(fname)[0]
                out_fname = f"{class_name}_{stem}.npy"
                out_path = os.path.join(config.PROCESSED_DATA_DIR, out_fname)
                np.save(out_path, mel)

                records.append({
                    "npy_path": out_path,
                    "label": label_map[class_name],
                    "class_name": class_name,
                })
            except Exception as e:
                print(f"[ERROR] {src_path}: {e}")

    manifest = pd.DataFrame(records)
    manifest_path = os.path.join(config.PROCESSED_DATA_DIR, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\nDone. {len(manifest)} samples saved.")
    print(f"Manifest: {manifest_path}")
    print(manifest["class_name"].value_counts())


if __name__ == "__main__":
    process_dataset()
