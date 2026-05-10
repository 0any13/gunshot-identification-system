"""
predict.py

Run the two-stage pipeline on any audio file.

Usage:
    docker compose run app python src/predict.py --file path/to/audio.wav
    docker compose run app python src/predict.py --file path/to/audio.wav --verbose
"""

import os
import argparse
import numpy as np
import librosa
import torch
import torch.nn.functional as F

import config
from model import get_stage1_model, get_stage2_model


def load_model(model, ckpt_name, device):
    path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def audio_to_mel(path):
    target_len = config.SAMPLE_RATE * config.CLIP_DURATION
    audio, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)

    if len(audio) >= target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")

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
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def predict(file_path, verbose=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = load_model(get_stage1_model(), "stage1_best.pth", device)
    stage2 = load_model(get_stage2_model(), "stage2_best.pth", device)

    mel = audio_to_mel(file_path).to(device)

    with torch.no_grad():
        # stage 1
        s1_logits = stage1(mel)
        s1_probs  = F.softmax(s1_logits, dim=1)[0]
        s1_pred   = s1_probs.argmax().item()
        s1_class  = config.STAGE1_CLASSES[s1_pred]
        s1_conf   = s1_probs[s1_pred].item()

        if verbose:
            print("\nStage 1 probabilities:")
            for cls, prob in zip(config.STAGE1_CLASSES, s1_probs):
                print(f"  {cls:<12}: {prob:.4f}")

        if s1_class == "siren":
            # stage 2
            s2_logits = stage2(mel)
            s2_probs  = F.softmax(s2_logits, dim=1)[0]
            s2_pred   = s2_probs.argmax().item()
            s2_conf   = s2_probs[s2_pred].item()
            s2_class  = (
                config.STAGE2_CLASSES[s2_pred]
                if s2_conf >= config.STAGE2_CONFIDENCE_THRESHOLD
                else "unknown"
            )

            if verbose:
                print("\nStage 2 probabilities:")
                for cls, prob in zip(config.STAGE2_CLASSES, s2_probs):
                    print(f"  {cls:<12}: {prob:.4f}")
                print(f"  confidence threshold: {config.STAGE2_CONFIDENCE_THRESHOLD}")

            final = f"siren ({s2_class})"
            final_conf = s2_conf
        else:
            final = s1_class
            final_conf = s1_conf

    print(f"\nFile   : {os.path.basename(file_path)}")
    print(f"Result : {final}")
    print(f"Confidence : {final_conf:.2%}")

    return final, final_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    required=True, help="Path to audio file (.wav, .mp3, etc.)")
    parser.add_argument("--verbose", action="store_true", help="Show all class probabilities")
    args = parser.parse_args()

    predict(args.file, verbose=args.verbose)