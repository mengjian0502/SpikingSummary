"""
VGGSNN of "TEMPORAL EFFICIENT TRAINING OF SPIKING NEURAL NETWORK VIA GRADIENT RE-WEIGHTING"
"""

import torch
import torch.nn as nn

class TETNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.AvgPool2d(2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.AvgPool2d(2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.AvgPool2d(2),
        )

        self.classifer1 = nn.Linear(4608, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifer1(x)
        return x

