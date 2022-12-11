"""
DVS-CIFAR10 model from "Direct Training for Spiking Neural Networks: Faster, Larger, Better"

AAAI-19
"""

import torch
import torch.nn as nn

class CIFARNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1),
            nn.Conv2d(128, 128, 3, stride=1),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, 3, stride=1),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.AvgPool2d(2),
        )

        self.classifer1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifer1(x)
        return x

