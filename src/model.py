"""
model.py

3-block CNN for audio classification on mel spectrograms.
Input shape: (batch, 1, N_MELS, time_frames)
Output shape: (batch, NUM_CLASSES)
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
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout=0.25),
            ConvBlock(32, 64, dropout=0.25),
            ConvBlock(64, 128, dropout=0.25),
        )

        # global average pooling collapses spatial dims regardless of input size
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


if __name__ == "__main__":
    # quick sanity check
    model = AudioCNN()
    dummy = torch.zeros(8, 1, config.N_MELS, 173)  # 173 time frames for 4s at hop=512
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect (8, NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
