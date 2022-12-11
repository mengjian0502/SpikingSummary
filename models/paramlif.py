"""
Model of ParamLIF

"""

import torch 
import torch.nn as nn

def create_conv_sequential(in_channels, out_channels, number_layer, use_max_pool):
    # 首层是in_channels-out_channels
    # 剩余number_layer - 1层都是out_channels-out_channels
    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
    ]

    for i in range(number_layer - 1):
        conv.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        ])
    return nn.Sequential(*conv)

def create_2fc(channels, h, w, class_num):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(channels * h * w, channels * h * w // 4, bias=False),
        nn.Linear(channels * h * w // 4, class_num * 10, bias=False),
    )

class PLIFNet(nn.Module):
    def __init__(self):
        super().__init__()
        in_channel = 2
        channels = 128
        w = 128
        h = 128

        self.conv = create_conv_sequential(in_channel, out_channels=channels, number_layer=4, use_max_pool=True)
        self.fc = create_2fc(channels=channels, w=w>>4, h=h>>4, class_num=10)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    