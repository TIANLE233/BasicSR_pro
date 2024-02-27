import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as f
from basicsr.archs.utils import Conv2d3x3, Upsampler


# @ARCH_REGISTRY.register()
class BicubicPP(nn.Module):
    def __init__(self, planes: int, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 act_layer: nn.Module = nn.LeakyReLU(inplace=True, negative_slope=0.1)) -> None:
        super(BicubicPP, self).__init__()
        
        self.down = nn.Sequential(nn.Conv2d(num_in_ch, planes, kernel_size=3, stride=2, padding=1, bias=False),
                                  act_layer)

        self.conv1 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                                   act_layer)
        self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                                   act_layer)

        self.tail = nn.Sequential(nn.Conv2d(planes, (2*upscale)**2 * num_out_ch, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2*upscale))

    def forward(self, x) -> torch.Tensor:

        down_x = self.down(x)

        conv1_x = self.conv1(down_x)
        conv2_x = self.conv2(conv1_x) + down_x

        x = self.tail(conv2_x)

        return x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    net = BicubicPP(planes=12, upscale=2, num_in_ch=4, num_out_ch=3, task='isp')
    print(count_parameters(net))

    data = torch.randn(1, 4, 224, 224)
    print(net(data).size())

