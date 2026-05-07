"""
augment.py

Audio augmentation applied BEFORE computing the mel spectrogram.
All functions take a numpy array (audio signal) and return a numpy array.

These are applied randomly during training to artificially expand the dataset
and improve generalisation to noisy real-world conditions.
"""

import numpy as np
import librosa
import random


def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def time_shift(audio, sr, max_shift_sec=0.5):
    max_shift = int(sr * max_shift_sec)
    shift = random.randint(-max_shift, max_shift)
    return np.roll(audio, shift)


def pitch_shift(audio, sr, n_steps_range=(-2, 2)):
    n_steps = random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def time_stretch(audio, rate_range=(0.85, 1.15)):
    rate = random.uniform(*rate_range)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    # re-pad or truncate to original length
    target = len(audio)
    if len(stretched) >= target:
        return stretched[:target]
    return np.pad(stretched, (0, target - len(stretched)), mode="constant")


def apply_random_augmentation(audio, sr, p=0.5):
    """
    Each augmentation is applied independently with probability p.
    Keep p around 0.5 so not every sample gets every transform.
    """
    if random.random() < p:
        audio = add_noise(audio)
    if random.random() < p:
        audio = time_shift(audio, sr)
    if random.random() < p:
        audio = pitch_shift(audio, sr)
    if random.random() < p:
        audio = time_stretch(audio)
    return audio
