import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as f
    
@ARCH_REGISTRY.register()
class sUNet(nn.Module):
    def __init__(self, planes: int, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 ) -> None:
        super(sUNet, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(num_in_ch, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        self.up = nn.Sequential(  nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=2),
                                  nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=2),
                                  nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=2)
                                )    
        self.conv1 = nn.Conv2d(planes, num_in_ch, kernel_size=(3, 3), padding=1)

        self.up0 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(num_in_ch, num_out_ch, kernel_size=(3, 3), padding=1)

    def forward(self, x) -> torch.Tensor:
        down_x = self.down(x)

        up_x = self.up(down_x)

        x = self.conv1(up_x) + self.up0(x)
        x = self.conv2(x)

        return x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # smallnet
    net = sUNet(planes=12, upscale=2, num_in_ch=4, num_out_ch=3, task='isp')
    print(count_parameters(net))

    data = torch.randn(1, 4, 224, 224)
    print(net(data).size())

