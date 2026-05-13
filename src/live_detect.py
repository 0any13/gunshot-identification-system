"""
live_detect.py

Real-time audio detection using the two-stage pipeline.
Runs directly on your machine (outside Docker) since it needs microphone access.

Setup:
    1. Open WO Mic on your phone and press Start
    2. Open WO Mic client on your PC and connect
    3. Run: python src/live_detect.py --list-devices
    4. Find "WO Mic Device" in the list and note its number
    5. Run: python src/live_detect.py --device <number>

Other options:
    python src/live_detect.py                        (uses default mic)
    python src/live_detect.py --threshold 0.90       (stricter alerts)
    python src/live_detect.py --interval 0.5         (check every 0.5s)
"""

import os
import sys
import argparse
import queue
import datetime
import numpy as np
import torch
import torch.nn.functional as F

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


def write_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    with open(log_filename, "a") as f:
        f.write(line + "\n")


def print_alert(result, conf):
    now = datetime.datetime.now().strftime("%I:%M:%S %p")

    if "gunshot" in result:
        label = "GUNSHOT"
    elif "siren" in result:
        siren_type = result.split("(")[-1].replace(")", "").strip().upper()
        label = f"{siren_type} SIREN"
    else:
        label = result.upper()

    line1 = f"  WARNING: {label} DETECTED at {now}"
    line2 = f"  Confidence: {conf:.2%}"
    width = max(len(line1), len(line2)) + 4

    print("\n" + "!" * width)
    print(line1)
    print(line2)
    print("!" * width + "\n")

    write_log(f"WARNING: {label} DETECTED at {now} | confidence: {conf:.2%}")


def print_clear():
    now = datetime.datetime.now().strftime("%I:%M:%S %p")
    msg = f"All clear at {now}"
    print(f"  [{msg}]")
    write_log(msg)


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_models(device):
    def load(model, ckpt_name):
        path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Make sure you run this from your project root:\n"
                "  cd C:\\Users\\oana\\Documents\\Projects\\DL\n"
                "  python src/live_detect.py"
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
    audio = audio_buffer[-target_len:].astype(np.float32)
    if len(audio) < target_len:
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


def run_inference(mel, stage1, stage2, device):
    mel = mel.to(device)
    with torch.no_grad():
        s1_probs = F.softmax(stage1(mel), dim=1)[0]
        s1_pred  = s1_probs.argmax().item()
        s1_class = config.STAGE1_CLASSES[s1_pred]
        s1_conf  = s1_probs[s1_pred].item()

        if s1_class == "siren":
            s2_probs = F.softmax(stage2(mel), dim=1)[0]
            s2_pred  = s2_probs.argmax().item()
            s2_conf  = s2_probs[s2_pred].item()
            s2_class = (
                config.STAGE2_CLASSES[s2_pred]
                if s2_conf >= config.STAGE2_CONFIDENCE_THRESHOLD
                else "unknown"
            )
            return f"siren ({s2_class})", s2_conf

        return s1_class, s1_conf


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def run(device_id, confidence_threshold, interval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    stage1, stage2 = load_models(device)

    print("\n" + "=" * 50)
    print("  GUNSHOT & SIREN DETECTION SYSTEM")
    print("=" * 50)
    print(f"  Mic device       : {'default' if device_id is None else device_id}")
    print(f"  Confidence limit : {confidence_threshold:.0%}")
    print(f"  Check every      : {interval}s")
    print(f"  Log file         : {log_filename}")
    print("=" * 50)
    print("\nListening... Press Ctrl+C to stop.\n")

    write_log("Session started")

    buffer_size  = config.SAMPLE_RATE * config.CLIP_DURATION
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

    last_result = "background"

    try:
        with sd.InputStream(**stream_kwargs):
            while True:
                try:
                    chunk = audio_queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                chunk_len = len(chunk)
                audio_buffer = np.roll(audio_buffer, -chunk_len)
                audio_buffer[-chunk_len:] = chunk

                result, conf = run_inference(
                    buffer_to_mel(audio_buffer), stage1, stage2, device
                )

                if result != "background" and conf >= confidence_threshold:
                    if result != last_result:
                        print_alert(result, conf)
                        last_result = result
                else:
                    if last_result != "background":
                        print_clear()
                        last_result = "background"

    except KeyboardInterrupt:
        write_log("Session ended by user")
        print("\nStopped. Log saved to:", log_filename)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",       type=int,   default=None,
                        help="Mic device index (use --list-devices to find it)")
    parser.add_argument("--threshold",    type=float, default=0.85,
                        help="Minimum confidence to trigger an alert (default: 0.85)")
    parser.add_argument("--interval",     type=float, default=0.5,
                        help="Seconds between inference runs (default: 0.5)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        sys.exit(0)

    run(
        device_id=args.device,
        confidence_threshold=args.threshold,
        interval=args.interval,
    )