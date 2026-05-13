"""
live_detect.py

Real-time audio detection using the two-stage pipeline.
Runs directly on your machine (outside Docker) since it needs microphone access.

Install dependency first:
    pip install sounddevice

Run:
    python src/live_detect.py
    python src/live_detect.py --threshold 0.85 --interval 0.5
    python src/live_detect.py --list-devices
    python src/live_detect.py --device 1
"""

import os
import sys
import argparse
import threading
import queue
import datetime
import numpy as np
import torch
import torch.nn.functional as F

# add src/ to path so config and model imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import get_stage1_model, get_stage2_model

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("librosa not installed. Run: pip install librosa")
    sys.exit(1)


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"detections_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)


def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_filename, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_models(device):
    def load(model, ckpt_name):
        path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Make sure you run this from your project root directory."
            )
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

    stage1 = load(get_stage1_model(), "stage1_best.pth")
    stage2 = load(get_stage2_model(), "stage2_best.pth")
    return stage1, stage2


# ---------------------------------------------------------------------------
# AUDIO PROCESSING
# ---------------------------------------------------------------------------

def buffer_to_mel(audio_buffer):
    target_len = config.SAMPLE_RATE * config.CLIP_DURATION

    if len(audio_buffer) >= target_len:
        audio = audio_buffer[-target_len:]
    else:
        audio = np.pad(audio_buffer, (0, target_len - len(audio_buffer)), mode="constant")

    audio = audio.astype(np.float32)

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


def run_inference(mel, stage1, stage2, device, confidence_threshold):
    mel = mel.to(device)

    with torch.no_grad():
        s1_logits = stage1(mel)
        s1_probs  = F.softmax(s1_logits, dim=1)[0]
        s1_pred   = s1_probs.argmax().item()
        s1_class  = config.STAGE1_CLASSES[s1_pred]
        s1_conf   = s1_probs[s1_pred].item()

        if s1_class == "siren":
            s2_logits = stage2(mel)
            s2_probs  = F.softmax(s2_logits, dim=1)[0]
            s2_pred   = s2_probs.argmax().item()
            s2_conf   = s2_probs[s2_pred].item()
            s2_class  = (
                config.STAGE2_CLASSES[s2_pred]
                if s2_conf >= config.STAGE2_CONFIDENCE_THRESHOLD
                else "unknown"
            )
            return f"siren ({s2_class})", s2_conf
        else:
            return s1_class, s1_conf


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def run(device_id=None, confidence_threshold=0.85, interval=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading models on {device}")
    stage1, stage2 = load_models(device)
    log("Models loaded. Listening...")
    log(f"Confidence threshold : {confidence_threshold:.0%}")
    log(f"Inference interval   : {interval}s")
    log(f"Log file             : {log_filename}")
    log("-" * 50)

    buffer_size = config.SAMPLE_RATE * config.CLIP_DURATION
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    audio_queue  = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        audio_queue.put(indata[:, 0].copy())

    stream_kwargs = dict(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(config.SAMPLE_RATE * interval),
        callback=audio_callback,
    )
    if device_id is not None:
        stream_kwargs["device"] = device_id

    last_result = None

    with sd.InputStream(**stream_kwargs):
        print("\nPress Ctrl+C to stop.\n")
        while True:
            try:
                chunk = audio_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            # roll buffer and append new chunk
            chunk_len = len(chunk)
            audio_buffer = np.roll(audio_buffer, -chunk_len)
            audio_buffer[-chunk_len:] = chunk

            mel    = buffer_to_mel(audio_buffer)
            result, conf = run_inference(mel, stage1, stage2, device, confidence_threshold)

            # only log when result changes or is a non-background detection
            if result != "background":
                if result != last_result:
                    log(f"DETECTED: {result.upper():<25} confidence: {conf:.2%}")
                    last_result = result
            else:
                if last_result is not None and last_result != "background":
                    log("back to background")
                last_result = "background"

            except KeyboardInterrupt:
                log("Stopped by user.")
                break


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold",    type=float, default=0.85,
                        help="Minimum confidence to log a detection (default: 0.85)")
    parser.add_argument("--interval",     type=float, default=0.5,
                        help="Seconds between inference runs (default: 0.5)")
    parser.add_argument("--device",       type=int,   default=None,
                        help="Microphone device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    run(
        device_id=args.device,
        confidence_threshold=args.threshold,
        interval=args.interval,
    )