"""
VGG7 model of ASF-BP
"""

import torch
import torch.nn as nn

class VGG7(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.AvgPool2d(2),
        )

        self.classifer1 = nn.Linear(128 * 8 * 8, 1024, bias=False)
        self.classifer2 = nn.Linear(1024, 10, bias=False)

    def forward(self, x):
        x = self.features(x)
        
        x = x.view(x.shape[0], -1)
        x = self.classifer1(x)
        x = self.classifer2(x)
        return x
