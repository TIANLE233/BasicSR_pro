import numpy as np
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as f

from basicsr.archs.utils import ESA, CCA


def extract_bayer_channels(raw):
    # Extract Bayer channels using array slicing
    raw = raw.squeeze(dim=1)
    ch_B = raw[:, 1::2, 1::2]
    ch_Gb = raw[:, 0::2, 1::2]
    ch_R = raw[:, 0::2, 0::2]
    ch_Gr = raw[:, 1::2, 0::2]

    # Combine channels into an RGB image
    RAW_combined = torch.stack((ch_B, ch_Gb, ch_R, ch_Gr), dim=1)
    return RAW_combined


class SAB(nn.Module):
    def __init__(self, planes: int, act_layer: nn.Module = nn.ReLU) -> None:
        super(SAB, self).__init__()

        self.esa = nn.Sequential(ESA(in_channels=planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.esa(x)


class CAB(nn.Module):
    def __init__(self, planes: int, act_layer: nn.Module = nn.ReLU) -> None:
        super(CAB, self).__init__()

        self.cca = nn.Sequential(CCA(planes=planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cca(x)


@ARCH_REGISTRY.register()
class MyNet(nn.Module):
    def __init__(self, planes: int, blocks: int, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 ) -> None:
        super(MyNet, self).__init__()
        # self.usf = nn.PixelUnshuffle(upscale)
        self.conv1 = nn.Sequential(nn.Conv2d(1, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(4, planes * 4, kernel_size=(3, 3), padding=1),
                                   nn.ReLU())
        self.sa = SAB(planes=planes * 4)
        self.ca = CAB(planes=planes)
        self.ps = nn.PixelShuffle(2)

        self.conv3 = nn.Sequential(nn.Conv2d(planes, num_out_ch, kernel_size=(3, 3), padding=1),
                                   nn.ReLU())
        # self.body = nn.Sequential(*[Block(planes=planes) for _ in range(blocks)])

        # self.tail = nn.Sequential(nn.Conv2d(planes, num_out_ch * (upscale ** 2), kernel_size=(3, 3), padding=1),
        #                           nn.ReLU(),
        #                           nn.PixelShuffle(upscale))

    def forward(self, x) -> torch.Tensor:
        packed_x = extract_bayer_channels(x)

        x1 = self.conv1(x)
        x2 = self.conv2(packed_x)

        x_sa = self.ca(x1)
        x_ca = self.ps(self.sa(x2))

        x_out = self.conv3(x_sa + x_ca)

        return x_out


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    net = MyNet(planes=8, blocks=1, upscale=2, num_in_ch=1, num_out_ch=3, task='isp')
    print(count_parameters(net))

    data = torch.randn(1, 1, 224, 224)
    print(net(data).size())
