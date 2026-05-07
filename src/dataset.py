"""
dataset.py

PyTorch Dataset that loads pre-computed mel spectrograms from .npy files.
The manifest CSV produced by preprocess.py is the index.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import config


class AudioDataset(Dataset):
    def __init__(self, manifest_path, augment=False):
        self.df = pd.read_csv(manifest_path)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["npy_path"])  # shape: (N_MELS, time_frames)

        # add channel dim so shape becomes (1, N_MELS, time_frames)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return mel, label


def get_dataloaders(manifest_path, batch_size=config.BATCH_SIZE, seed=42):
    full_dataset = AudioDataset(manifest_path, augment=False)
    n = len(full_dataset)

    n_train = int(n * config.TRAIN_SPLIT)
    n_val = int(n * config.VAL_SPLIT)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # enable augmentation flag on train subset's underlying dataset
    # note: random_split wraps, so augmentation is set at the Dataset level
    # for simplicity augmentation is currently applied in train.py via transforms

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Dataset split -> train: {n_train} | val: {n_val} | test: {n_test}")
    return train_loader, val_loader, test_loader
