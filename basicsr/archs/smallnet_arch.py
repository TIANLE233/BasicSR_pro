import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
# from archs.efdn_arch import EDBB
# from archs.fmen_arch import RRRB
import torch.nn.functional as f


class Block(nn.Module):
    def __init__(self, planes: int, act_layer: nn.Module = nn.ReLU) -> None:
        super(Block, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
                                  act_layer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) 

@ARCH_REGISTRY.register()
class Smallnet(nn.Module):
    def __init__(self, planes: int, blocks: int, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 ) -> None:
        super(Smallnet, self).__init__()
        # 第一层卷积，输入通道4，输出通道16，使用Tanh激活函数
        self.head = nn.Sequential(nn.Conv2d(num_in_ch, planes, kernel_size=(3, 3), padding=1),
                                  nn.ReLU())

        # 第二层卷积，输入通道16，输出通道16，使用ReLU激活函数
        # self.body = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
        #                            nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[Block(planes=planes) for _ in range(blocks)])

        # 第三层卷积，输入通道16，输出通道12，使用ReLU激活函数
        self.tail = nn.Sequential(nn.Conv2d(planes, num_out_ch * (upscale**2), kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),\
                                  nn.PixelShuffle(upscale))

        # PixelShuffle层将通道映射到RGB
        

    def forward(self, x) -> torch.Tensor:
        # 第一层卷积，Tanh激活
        head_x = self.head(x)

        # x = self.sconv(x)

        # 第二层卷积，ReLU激活
        body_x = self.body(head_x)
        
        # body_x = body_x + head_x

        # 第三层卷积，ReLU激活
        x = self.tail(body_x)

        return x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # smallnet
    net = Smallnet(planes=12, upscale=2, num_in_ch=4, num_out_ch=3, task='isp')
    print(count_parameters(net))

    data = torch.randn(1, 4, 224, 224)
    print(net(data).size())

