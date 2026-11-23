"""CNN trained from scratch (model_name = "cnn_scratch").

Same building blocks as in class: Conv -> BatchNorm -> ReLU, MaxPool to halve the size,
more filters as the image gets smaller, then a small classifier with Dropout.

Input 3 x 224 x 224  ->  4 blocks (32, 64, 128, 256 filters)  ->  256 x 14 x 14
->  global average pooling  ->  256 numbers  ->  Dropout  ->  Linear  ->  C logits

Global average pooling (instead of Flatten) keeps the model small (~1.2M parameters;
Flatten would add 256*14*14*56 = 2.8M weights in the last layer alone), so it overfits
less on 33k images, and it accepts any input size (224x224 or 224x448 "wide").
"""
from __future__ import annotations

import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions, then halve height and width."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers.append(conv_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

    def gradcam_layer(self):
        """Last conv block - the layer Maya should hook for Grad-CAM."""
        return self.features[-1]
