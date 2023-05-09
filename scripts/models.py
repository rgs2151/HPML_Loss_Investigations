import torch
from torch import nn, Tensor
from torch.nn import Module

import torchvision
from torchvision.models import UNet


torchvision.models.unet.norm_type = 'GroupNorm'


class ExampleModel1(Module):
    def __init__(self) -> None:
        super().__init__()
        self.cunet = UNet(4, 4, [32, 64, 128, 256, 512], True)
        self.pad = nn.ConstantPad2d((11, 12, 7, 7), 0)

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        t = self.pad(input)
        t = self.cunet(t)
        t = t[:, :, 7:-7, 11:-12]
        output = identity + t
        return output

class ExampleModel2(Module):
    def __init__(self) -> None:
        super().__init__()
        self.cunet = UNet(4, 4, [32, 64, 128, 256, 512], True)
        self.pad = nn.ConstantPad2d((11, 12, 7, 7), 0)

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        t = self.pad(input)
        t = self.cunet(t)
        t = t[:, :, 7:-7, 11:-12]
        output = identity + t
        return output

class ExampleModel3(Module):
    def __init__(self) -> None:
        super().__init__()
        self.cunet = UNet(4, 4, [32, 64, 128, 256, 512], True)
        self.pad = nn.ConstantPad2d((11, 12, 7, 7), 0)

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        t = self.pad(input)
        t = self.cunet(t)
        t = t[:, :, 7:-7, 11:-12]
        output = identity + t
        return output

class ExampleModel4(Module):
    def __init__(self) -> None:
        super().__init__()
        self.cunet = UNet(4, 4, [32, 64, 128, 256, 512], True)
        self.pad = nn.ConstantPad2d((11, 12, 7, 7), 0)

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        t = self.pad(input)
        t = self.cunet(t)
        t = t[:, :, 7:-7, 11:-12]
        output = identity + t
        return output