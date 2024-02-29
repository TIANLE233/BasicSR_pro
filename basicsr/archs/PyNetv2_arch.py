import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyNETv2(nn.Module):
    def __init__(self):
        super(PyNETv2, self).__init__()

    def forward(self, x) -> torch.Tensor:
        # 第一层卷积，Tanh激活
        x = PyNET(x)

        return x

def PyNET(input, i_n=False, instance_norm_level_1=False):
    k = 2

    x = input

    conv_l1_d1 = conv_layer(x, 32, 3, 1, relu=True, i_n=False, padding='SAME')
    conv_l2_d1 = conv_layer(conv_l1_d1, 64, 2, 2, relu=True, i_n=False, padding='VALID')
    conv_l3_d1 = conv_layer(conv_l2_d1, 128, 2, 2, relu=True, i_n=False, padding='VALID')

    conv_l3_d6 = residual_groups_residual(conv_l3_d1, 128, i_n=True, groups=4, n=2)
    conv_l3_d8 = sam_block(conv_l3_d6, 128, i_n=False) + conv_l3_d6
    conv_l3_d9 = cam_block(conv_l3_d8, 128, i_n=False) + conv_l3_d8

    # -> Output: Level 3

    conv_l3_out = conv_layer(conv_l3_d9, 3 * k * k, 1, 1, relu=False,
                             i_n=False)  # 32 -> 128  # 128ch -> 48ch
    conv_l3_out = nn.PixelShuffle(k)(conv_l3_out)
    output_l3 = torch.tanh(conv_l3_out) * 0.58 + 0.5

    conv_t2a = upsample_layer(conv_l3_d9, 64, 2, 2)
    conv_l2_d2 = conv_layer(conv_l2_d1, 64, 3, 1, relu=True, i_n=False, padding='SAME')

    conv_l2_d3 = conv_l2_d2 + conv_t2a

    conv_l2_d12 = conv_residual_1x1(conv_l2_d3, 64, i_n=False)
    conv_l2_d13 = residual_groups_residual(conv_l2_d12, 64, i_n=True, groups=2, n=3)
    conv_l2_d16 = cam_block(conv_l2_d13, 64, i_n=False) + conv_l2_d13

    # -> Output: Level 2

    conv_l2_out = conv_layer(conv_l2_d16, 3 * k * k, 1, 1, relu=False, i_n=False)
    conv_l2_out = nn.PixelShuffle(k)(conv_l2_out)
    output_l2 = torch.tanh(conv_l2_out) * 0.58 + 0.5

    conv_t1a = upsample_layer(conv_l2_d16, 32, 2, 2)
    conv_l1_d2 = conv_layer(conv_l1_d1, 32, 3, 1, relu=True, i_n=False, padding='SAME')

    conv_l1_d3 = conv_l1_d2 + conv_t1a

    conv_l1_d12 = conv_residual_1x1(conv_l1_d3, 32, i_n=False)
    conv_l1_d13 = residual_groups_residual(conv_l1_d12, 32, i_n=True, groups=4)
    conv_l1_d14 = residual_groups_residual(conv_l1_d13, 32, i_n=True, groups=2)

    # -> Output: Level 1
    conv_l1_out = conv_layer(conv_l1_d14, 4 * k * k, 1, 1, relu=True, i_n=False)
    conv_l1_out = nn.PixelShuffle(k)(conv_l1_out)
    conv_l1_out = conv_layer(conv_l1_out, 3, 3, 1, relu=False, i_n=False)
    output_l1 = torch.tanh(conv_l1_out) * 0.58 + 0.5

    # return None, output_l1, output_l2, output_l3
    return output_l1


def conv_residual_1x1(input, num_maps, i_n):
    conv_3a = conv_layer(input, num_maps, 1, 1, relu=False, i_n=i_n)

    output_tensor = nn.PReLU()(conv_3a) + input

    return output_tensor


def residual_groups_residual(input, num_maps, i_n, groups=4, n=1):
    groups_tf = []

    batch, rows, cols, channels = [i for i in input.get_shape()]

    step = int(channels) // groups
    assert (int(channels) % groups == 0)

    for i in range(groups):
        groups_tf.append(input[:, :, :, step * i:step * (i + 1)])

    values = []
    for i, g in enumerate(groups_tf):
        for k in range(n):
            g = conv_layer(g, num_maps // groups, 3, 1, relu=True, i_n=(i_n and (i % 2) == 0)) + g
        values.append(g)

    conv_3a = torch.cat(values, dim=1)

    conv_3a = conv_layer(conv_3a, num_maps, 1, 1, relu=False, i_n=False)

    output_tensor = conv_3a + input

    return output_tensor


# def _sam_block(input, num_maps, instance_norm):
#     out = _conv_layer(input, num_maps, 3, 1, relu=True, instance_norm=instance_norm)
#
#     spatial_att = _dw_conv_layer(out, num_maps, 5, 1, relu=False, instance_norm=False)
#     spatial_att = tf.math.sigmoid(spatial_att)
#
#     return spatial_att * out
def sam_block(input, num_maps, i_n):
    out = conv_layer(input, num_maps, 3, 1, relu=True, i_n=i_n)

    spatial_att = dw_conv_layer(out, num_maps, 5, 1, relu=False, i_n=False)
    spatial_att = nn.Sigmoid()(spatial_att)

    return spatial_att * out


# def _cam_block(input, num_maps, i_n=False):
#     out = conv_layer(input, num_maps, 3, 1, relu=True, i_n=i_n)
#
#     channel_att = conv_layer(out, num_maps, 1, 3, relu=True, i_n=False, padding='VALID')
#     channel_att = conv_layer(channel_att, num_maps, 3, 3, relu=True, i_n=False, padding='VALID')
#     channel_att = tf.math.reduce_mean(channel_att, axis=[1, 2], keepdims=True, name=None)
#     channel_att = conv_layer(channel_att, num_maps, 1, 1, relu=True, i_n=False, padding='VALID')
#     channel_att = conv_layer(channel_att, num_maps, 1, 1, relu=False, i_n=False, padding='VALID')
#     channel_att = tf.math.sigmoid(channel_att)
#
#     return channel_att * out
#
#
# def stack(x, y):
#     return tf.concat([x, y], -1)
def cam_block(input, num_maps, i_n=False):
    out = conv_layer(input, num_maps, 3, 1, relu=True, i_n=i_n)

    channel_att = conv_layer(out, num_maps, 1, 3, relu=True, i_n=False, padding='VALID')
    channel_att = conv_layer(channel_att, num_maps, 3, 3, relu=True, i_n=False, padding='VALID')
    channel_att = torch.mean(channel_att, dim=[2, 3], keepdim=True)
    channel_att = conv_layer(channel_att, num_maps, 3, 3, relu=True, i_n=False, padding='VALID')
    channel_att = conv_layer(channel_att, num_maps, 3, 3, relu=True, i_n=False, padding='VALID')
    channel_att = nn.Sigmoid()(channel_att)

    # 将通道注意力权重应用到原始特征图上

    return channel_att * out


# def _dw_conv_layer(net, num_filters, filter_size, strides, relu=True, i_n=False, padding='SAME'):
#     if filter_size // 2 >= 1:
#         paddings = tf.constant(
#             [[0, 0], [filter_size // 2, filter_size // 2], [filter_size // 2, filter_size // 2], [0, 0]])
#         net = tf.pad(net, paddings, mode='REFLECT')
#
#     net = tf.keras.layers.DepthwiseConv2D(
#         filter_size, strides=(strides, strides), padding='valid', depth_multiplier=1,
#         data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
#         depthwise_initializer='glorot_uniform',
#         bias_initializer='zeros', depthwise_regularizer=None,
#         bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None,
#         bias_constraint=None
#     )(net)
#
#     if i_n:
#         net = instance_norm(net)
#
#     if relu:
#         net = tf.keras.layers.PReLU(shared_axes=[1, 2])(net)
#
#     return net
def dw_conv_layer(net, num_filters, filter_size, strides, relu=True, i_n=False, padding='SAME'):
    # Apply padding if necessary to match 'SAME' padding in TensorFlow

    if padding == 'SAME' and filter_size % 2 == 1:
        padding_height = padding_width = filter_size // 2

        net = nn.functional.pad(net, (padding_width, padding_width, padding_height, padding_height), mode='reflect')

        # Create the depthwise convolution layer

    dwconvlayer = nn.Conv2d(
        in_channels=net.shape[1],
        out_channels=num_filters,
        kernel_size=filter_size,
        stride=strides,
        groups=net.shape[1],
        padding=0 if padding == 'VALID' else padding_height,  # Use 0 for 'VALID' padding
        bias=True
    )

    # Apply the depthwise convolution

    net = dwconvlayer(net)

    if i_n:
        net = instance_norm(net)

    if relu:
        net = nn.PReLU()(net)

    return net


# def _conv_layer(net, num_filters, filter_size, strides, relu=True, i_n=False, padding='SAME'):
#     if filter_size // 2 >= 1 and padding == 'SAME':
#         paddings = tf.constant(
#             [[0, 0], [filter_size // 2, filter_size // 2], [filter_size // 2, filter_size // 2], [0, 0]])
#         net = tf.pad(net, paddings, mode='REFLECT')
#
#     net = tf.keras.layers.Conv2D(
#         num_filters, filter_size, strides=(strides, strides), padding='valid',
#         data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
#         use_bias=True, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=1),
#         bias_initializer='zeros', kernel_regularizer=None,
#         bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
#         bias_constraint=None)(net)
#
#     if i_n:
#         net = instance_norm(net)
#
#     if relu:
#         net = tf.keras.layers.PReLU(shared_axes=[1, 2])(net)
#
#     return net
def conv_layer(net, num_filters, filter_size, strides, relu=True, i_n=False, padding='SAME'):
    # Apply padding if necessary to match 'SAME' padding in TensorFlow

    if padding == 'SAME' and filter_size % 2 == 1:
        padding_height = padding_width = filter_size // 2

        net = nn.functional.pad(net, (padding_width, padding_width, padding_height, padding_height), mode='reflect')

        # Create the convolution layer

    convlayer = nn.Conv2d(
        in_channels=net.shape[1],
        out_channels=num_filters,
        kernel_size=filter_size,
        stride=strides,
        padding=0 if padding == 'VALID' else padding_height,  # Use 0 for 'VALID' padding
        bias=True
    )
    net = convlayer(net)

    # Apply instance normalization if requested

    if i_n:
        net = instance_norm(net)

        # Apply PReLU activation if requested

    if relu:
        net = nn.PReLU()(net)

    return net


# def _instance_norm(net):
#     return tfa.layers.InstanceNormalization()(net)
def instance_norm(net):
    # 创建实例归一化层
    i_n = nn.InstanceNorm2d(net.size()[1])  # 输入通道数
    # 应用实例归一化
    net = i_n(net)

    return net


# def _conv_tranpose_layer(net, num_filters, filter_size, strides, relu=True):
#     net = tf.keras.layers.Conv2DTranspose(
#         num_filters, filter_size, strides=(strides, strides), padding='same',
#         output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None,
#         use_bias=True, kernel_initializer='glorot_uniform',
#         bias_initializer='zeros', kernel_regularizer=None,
#         bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
#         bias_constraint=None
#     )(net)
#
#     if relu:
#         return tf.keras.layers.PReLU(shared_axes=[1, 2])(net)
#     else:
#         return net
def conv_transpose_layer(net, num_filters, filter_size, strides, relu=True):
    net = nn.ConvTranspose2d(
        in_channels=net.shape[1],
        out_channels=num_filters,
        kernel_size=filter_size,
        stride=strides,
        padding=0,  # 'same' padding in TensorFlow is achieved by adjusting the output padding
        output_padding=(filter_size - 1) * (strides - 1),  # Adjust output padding for 'same' behavior
        bias=True,
        dilation=1,
        groups=1
    )(net)

    if relu:
        net = nn.PReLU()(net)

    return net


# def _upsample_layer(net, num_filters, filter_size, strides, relu=True):
#     net = tf.keras.layers.UpSampling2D(
#         size=(strides, strides), data_format=None, interpolation='bilinear',
#     )(net)
#
#     paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
#     net = tf.pad(net, paddings, mode='REFLECT')
#
#     net = tf.keras.layers.Conv2D(
#         num_filters, 3, strides=(1, 1), padding='valid',
#         data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
#         use_bias=True, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=1),
#         bias_initializer='zeros', kernel_regularizer=None,
#         bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
#         bias_constraint=None)(net)
#
#     if relu:
#         return tf.keras.layers.PReLU(shared_axes=[1, 2])(net)
#     else:
#         return net

def upsample_layer(net, num_filters, filter_size, strides, relu=True):
    net = F.interpolate(net, scale_factor=strides, mode='bilinear', align_corners=False)

    # Padding in PyTorch is applied directly during the convolution,

    # so we don't need to create a separate padding layer.

    # We'll adjust the padding in the convolution layer instead.

    net = nn.Conv2d(
        in_channels=net.shape[1],
        out_channels=num_filters,
        kernel_size=3,
        stride=1,
        padding=1,  # Adjusted padding to match the TensorFlow version's effective padding
        bias=True,
        dilation=1,
        groups=1
    )(net)

    if relu:
        net = nn.PReLU()(net)

    return net


# def max_pool(x, n):
#     return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def max_pool(x, n):
    # PyTorch中的MaxPool2d期望输入是(batch_size, channels, height, width)
    # 如果x的维度不是这样的，您可能需要调整它
    return nn.functional.max_pool2d(x, kernel_size=n, stride=n)

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # smallnet
    net = PyNETv2()
    print(count_parameters(net))

