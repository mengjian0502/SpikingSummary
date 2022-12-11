"""
Get model details
"""

import torch
import torch.nn as nn
from models import CIFARNet, VGG7, resnet19, PLIFNet, BPSANet, SEWResNet, vgg11_lif, TETNet

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    x = torch.randn(1, 3, 128, 128)
    num_classes = 10
    precision=32
    wbit = 32

    # model = CIFARNet(num_classes=num_classes)
    # model = resnet19(num_classes=num_classes)
    # model = PLIFNet()
    # model = SEWResNet(connect_f='ADD')
    # model = TETNet(None)
    model = BPSANet(num_classes=num_classes)

    nparam = 0
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d)):
            if m.weight.size(2) > 1:
                m.register_forward_hook(get_activation(n))
                nparam += m.weight.numel()
        elif isinstance(m, (nn.Linear)):
                m.register_forward_hook(get_activation(n))
                nparam += m.weight.numel()
            

    
    y = model(x)
    
    total_pixels = 0
    for k, v in activation.items():
        total_pixels += v.numel()
    
    print("Total pixels = {}".format(total_pixels))
    print("Total Membrane = {} (MB)".format(total_pixels*precision/8/1e+6))
    
    print("Total weights = {}".format(nparam))
    print("Total weights storage = {} (MB)".format(nparam*wbit/8/1e+6))

if __name__ == '__main__':
    main()