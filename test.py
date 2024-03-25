import numpy as np
import torch
import torch.nn.functional as F


def extract_bayer_channels(raw):
    # Extract Bayer channels using array slicing
    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    # Combine channels into an RGB image
    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    print(ch_B)
    return RAW_combined


def bayer2rggb(raw):
    h, w = raw.shape
    raw = raw.reshape(h // 2, 2, w // 2, 2)
    raw = raw.transpose([1, 3, 0, 2]).reshape([-1, h // 2, w // 2])
    return raw


def bayer24(input_data):
    # 进行卷积操作，不使用偏置
    weight = torch.tensor([[[[1, 0], [0, 0]]],
                           [[[0, 1], [0, 0]]],
                           [[[0, 0], [1, 0]]],
                           [[[0, 0], [0, 1]]]], dtype=torch.float32)  # 卷积核为一个大小为 (out_channels, in_channels, kernel_height, kernel_width) 的四维张量

    output_data = F.conv2d(input_data, weight, bias=None, stride=(2,2), padding=(0,0))
    return output_data


# 自定义卷积核参数
# 创建自定义输入数据
input_data = torch.tensor([[[[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10],
                             [11, 12, 13, 14, 15],
                             [16, 17, 18, 19, 20],
                             [21, 22, 23, 24, 25]]]])  # 输入数据为一个大小为 (batch_size, channels, height, width) 的四维张量


input_matrix = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                         [9, 10, 11, 12, 13, 14, 15, 16],
                         [17, 18, 19, 20, 21, 22, 23, 24],
                         [25, 26, 27, 28, 29, 30, 31, 32],
                         [33, 34, 35, 36, 37, 38, 39, 40],
                         [41, 42, 43, 44, 45, 46, 47, 48],
                         [49, 50, 51, 52, 53, 54, 55, 56],
                         [57, 58, 59, 60, 61, 62, 63, 64]])

# 调用函数并输出结果
bayer_channels = extract_bayer_channels(input_matrix)
rggb_image = bayer2rggb(input_matrix)
output_data = custom_conv2d(torch.unsqueeze(torch.tensor(input_matrix, dtype=torch.float32), dim=0))

print("bayer_channels:\n", bayer_channels)
print("\nrggb_image:\n", rggb_image)
print("\noutput_data:\n", output_data)
