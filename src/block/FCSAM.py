# This code is released under the CC BY-SA 4.0 license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fca import MultiSpectralAttentionLayer

# 单头自注意力（包含dct部分）:先对Input做MSA，然后再用sa的q,k,v卷积DCT+SA

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class FCSA(nn.Module):
    def __init__(self, in_channels, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 16]):
        super().__init__()
        self.in_channels = in_channels
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)
        self.att_MSA = MultiSpectralAttentionLayer(in_channels, c2wh[in_channels//4], c2wh[in_channels//4], reduction=16,
                                                   freq_sel_method='top16')

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # .view(-1)是根据另外的数自动调整维度
        # input: B, C, H, W -> q: B, H * W, C // 8
        input_dct = self.att_MSA(input)
        q = self.query(input_dct).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(input_dct).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input_dct).view(batch_size, -1, height * width)

        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.

        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        out = F.leaky_relu(out)

        return self.gamma * out + input
