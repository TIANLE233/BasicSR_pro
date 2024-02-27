import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.utils import Conv2d3x3, Upsampler


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 上采样
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


@ARCH_REGISTRY.register()
class UNet(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 ) -> None:
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.head = ConvBlock(ch_in=num_in_ch, ch_out=64)
        # 编码器中的卷积块
        self.encoders = nn.ModuleList([
            ConvBlock(ch_in=64, ch_out=128),
            ConvBlock(ch_in=128, ch_out=256),
            ConvBlock(ch_in=256, ch_out=512),
            ConvBlock(ch_in=512, ch_out=1024)
        ])

        # 解码器中的上采样卷积和卷积块
        self.upconvs = nn.ModuleList([
            UpConv(ch_in=1024, ch_out=512),
            UpConv(ch_in=512, ch_out=256),
            UpConv(ch_in=256, ch_out=128),
            UpConv(ch_in=128, ch_out=64)
        ])

        self.up_conv_blocks = nn.ModuleList([
            ConvBlock(ch_in=1024, ch_out=512),
            ConvBlock(ch_in=512, ch_out=256),
            ConvBlock(ch_in=256, ch_out=128),
            ConvBlock(ch_in=128, ch_out=64)
        ])

        # 最终输出的1x1卷积层
        self.tail = nn.Sequential(Upsampler(upscale=upscale, in_channels=64,
                                            out_channels=64, upsample_mode=task),
                                  Conv2d3x3(64, num_out_ch))

    def forward(self, x):
        encoder_outs = []
        x = self.head(x)
        encoder_outs.append(x)
        for encoder in self.encoders:
            x = self.Maxpool(x)
            x = encoder(x)
            encoder_outs.append(x)

        x = encoder_outs.pop()
        for upconv, up_conv_block in zip(self.upconvs, self.up_conv_blocks):
            x = upconv(x)
            x = torch.cat((encoder_outs.pop(), x), dim=1)
            x = up_conv_block(x)

        # 使用PixelShuffle将通道映射到RGB
        x = self.tail(x)
        return x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # UNet
    net = UNet(upscale=2, num_in_ch=4, num_out_ch=3, task='lsr')
    print(count_parameters(net))

    data = torch.randn(1, 4, 224, 224)
    print(net(data).size())
