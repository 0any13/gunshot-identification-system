"""
model.py

Two CNN models sharing the same architecture but different output sizes:
- AudioCNN(num_classes=3) for Stage 1 (gunshot / siren / background)
- AudioCNN(num_classes=4) for Stage 2 (police / ambulance / firetruck / unknown)
"""

import torch
import torch.nn as nn
import config


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32,  dropout=0.25),
            ConvBlock(32, 64, dropout=0.25),
            ConvBlock(64, 128, dropout=0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def get_stage1_model():
    return AudioCNN(num_classes=config.NUM_STAGE1_CLASSES)


def get_stage2_model():
    return AudioCNN(num_classes=config.NUM_STAGE2_CLASSES)


if __name__ == "__main__":
    dummy = torch.zeros(8, 1, config.N_MELS, 173)

    m1 = get_stage1_model()
    out1 = m1(dummy)
    print(f"Stage 1 output: {out1.shape}")

    m2 = get_stage2_model()
    out2 = m2(dummy)
    print(f"Stage 2 output: {out2.shape}")

    params = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f"Parameters per model: {params:,}")