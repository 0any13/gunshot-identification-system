"""
dataset.py

Dataset and DataLoader builder for both stages.
UrbanSound8K files are prefixed with us8k_{fsID}_ and are grouped
by fsID to prevent data leakage across splits.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit

import config


class AudioDataset(Dataset):
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["npy_path"])
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return mel, label


def _get_groups(df):
    groups = []
    us8k_pattern = re.compile(r"us8k_(\d+)_")
    unique_counter = 10_000_000

    for _, row in df.iterrows():
        fname = os.path.basename(row["npy_path"])
        match = us8k_pattern.search(fname)
        if match:
            groups.append(int(match.group(1)))
        else:
            groups.append(unique_counter)
            unique_counter += 1

    return groups


def get_dataloaders(manifest_path, batch_size=config.BATCH_SIZE, seed=42):
    dataset = AudioDataset(manifest_path)
    df = dataset.df
    groups = _get_groups(df)
    all_indices = list(range(len(df)))

    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=1 - config.TRAIN_SPLIT - config.VAL_SPLIT,
        random_state=seed,
    )
    trainval_idx, test_idx = next(gss_test.split(all_indices, groups=groups))

    trainval_groups = [groups[i] for i in trainval_idx]
    val_ratio = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    rel_train_idx, rel_val_idx = next(gss_val.split(trainval_idx, groups=trainval_groups))

    train_idx = [trainval_idx[i] for i in rel_train_idx]
    val_idx   = [trainval_idx[i] for i in rel_val_idx]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Split -> train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")
    return train_loader, val_loader, test_loader