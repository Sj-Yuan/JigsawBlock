# This code is released under the CC BY-SA 4.0 license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from atten.fca import MultiSpectralAttentionLayer
from .SE_weight_module import SEWeightModule


# 单头自注意力（包含dct部分）:先对Input做MSA，然后再用sa的q,k,v卷积,跑tile很好

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class mfcsa_module_layer_2(nn.Module):
    # 分组卷积(conv_group)相当于稀疏卷积，意义就是减少参数量，多尺度卷积核会导致参数量上升，group操作可以减少参数
    # group这个操作在alexnet就有 conv_groups=[1, 4, 8, 16]
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(mfcsa_module_layer_2, self).__init__()
        self.in_channels = inplans
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.conv_1 = conv(inplans, planes//2, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//2, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        # self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
        #                     stride=stride, groups=conv_groups[2])
        # self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
        #                     stride=stride, groups=conv_groups[3])
        # 这里inplans//4的时候可以跑通
        self.att_MSA = MultiSpectralAttentionLayer(inplans//2, c2wh[planes//2], c2wh[planes//2],
                                                   reduction=16,
                                                   freq_sel_method='top16')
        self.selfatten = selfattention(inplans//2)

        self.split_channel = planes // 2
        self.softmax = nn.Softmax(dim=1)

    # x=[4,256,64,64], x1=[4,64,64,64],x2=[4,64,64,64]
    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        # x3 = self.conv_3(x)
        # x4 = self.conv_4(x)

        feats = torch.cat((x1, x2), dim=1)
        feats = feats.view(batch_size, 2, self.split_channel, feats.shape[2], feats.shape[3])
        # 求每个scale的channel_atten
        x1_msa = self.selfatten(x1)
        x2_msa = self.selfatten(x2)
        # x3_msa = self.selfatten(x3)
        # x4_msa = self.selfatten(x4)
        # x_se=[4,256,64,64], x1_msa=[4,64,64,64],x2_msa=[4,64,64,64]

        x_msa = torch.cat((x1_msa, x2_msa), dim=1)
        attention_vectors = x_msa.view(batch_size, 2, self.split_channel, x_msa.shape[2], x_msa.shape[3])
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors

        for i in range(2):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

class mfcsa_module_layer_4(nn.Module):
    # 分组卷积(conv_group)相当于稀疏卷积，意义就是减少参数量，多尺度卷积核会导致参数量上升，group操作可以减少参数
    # group这个操作在alexnet就有 conv_groups=[1, 4, 8, 16]
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(mfcsa_module_layer_4, self).__init__()
        self.in_channels = inplans
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        # 这里inplans//4的时候可以跑通
        self.att_MSA = MultiSpectralAttentionLayer(inplans//4, c2wh[planes//4], c2wh[planes//4],
                                                   reduction=16,
                                                   freq_sel_method='top16')
        self.selfatten = selfattention(inplans//4)

        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    # x=[4,256,64,64], x1=[4,64,64,64],x2=[4,64,64,64]
    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
        # 求每个scale的channel_atten
        x1_msa = self.selfatten(x1)
        x2_msa = self.selfatten(x2)
        x3_msa = self.selfatten(x3)
        x4_msa = self.selfatten(x4)
        # x_se=[4,256,64,64], x1_msa=[4,64,64,64],x2_msa=[4,64,64,64]

        x_msa = torch.cat((x1_msa, x2_msa, x3_msa, x4_msa), dim=1)
        attention_vectors = x_msa.view(batch_size, 4, self.split_channel, x_msa.shape[2], x_msa.shape[3])
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors

        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class selfattention(nn.Module):
    def __init__(self, in_channels, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 16]):
        super().__init__()
        self.in_channels = in_channels
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)
        self.att_MSA = MultiSpectralAttentionLayer(in_channels, c2wh[in_channels], c2wh[in_channels], reduction=16,
                                                   freq_sel_method='top16')

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # .view(-1)是根据另外的数自动调整维度
        # input: B, C, H, W -> q: B, H * W, C // 8
        # q = self.query(input)
        input_dct = self.att_MSA(input)
        q = self.query(input_dct).view(batch_size, -1, height * width).permute(0, 2, 1)
        # 先对Input做dct操作，获得dct后的矩阵，input: B, C, H, W -> input_dct: B, H * W, C // 8
        # input_dct = self.att_MSA(input)
        # q_dct = self.att_MSA(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        # k = self.key(input)

        k = self.key(input_dct).view(batch_size, -1, height * width)
        # 不同的att_msa对Input的运算值不同，同一个att_msa对input的运算值相同

        # q_dct = self.att_MSA(q).view(batch_size, -1, height * width).permute(0, 2, 1)
        # q_se = self.att_se(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # k_dct = self.att_MSA(k).view(batch_size, -1, height * width)
        # k_se = self.att_se(input).view(batch_size, -1, height * width)

        # input: B, C, H, W -> v: B, C, H * W
        # v = self.value(input)
        v = self.value(input_dct).view(batch_size, -1, height * width)

        # 对input做dct
        # v_dct = self.att_MSA(v).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        # attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.

        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        # out_dct = torch.bmm(v_dct, attn_matrix.permute(0, 2, 1))  # 用dct后的v计算
        out = out.view(*input.shape)

        # 尝试加入leaky_relu
        # ======================================
        out = F.leaky_relu(out)
        # ======================================


        return self.gamma * out + input
        # return out

# 跑carpet没问题
# class selfattention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
#         self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
#         self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
#         self.softmax = nn.Softmax(dim=-1)
#         self.att_MSA = MultiSpectralAttentionLayer(in_channels, c2wh[in_channels//4], c2wh[in_channels//4], reduction=16,
#                                                    freq_sel_method='top16')
#
#     def forward(self, input):
#         batch_size, channels, height, width = input.shape
#         # .view(-1)是根据另外的数自动调整维度
#         # input: B, C, H, W -> q: B, H * W, C // 8
#         #q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
#         q_dct = self.att_MSA(input).view(batch_size, -1, height * width).permute(0, 2, 1)
#         # input: B, C, H, W -> k: B, C // 8, H * W
#         #k = self.key(input).view(batch_size, -1, height * width)
#         k_dct = self.att_MSA(input).view(batch_size, -1, height * width)
#
#         # input: B, C, H, W -> v: B, C, H * W
#         #v = self.value(input).view(batch_size, -1, height * width)
#         v_dct = self.att_MSA(input).view(batch_size, -1, height * width)
#         # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
#         attn_matrix = torch.bmm(q_dct, k_dct)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
#         #out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
#         out = torch.bmm(v_dct, attn_matrix.permute(0, 2, 1))  # 用dct后的v计算
#         out = out.view(*input.shape)
#
#         # 尝试加入leaky_relu
#         # ======================================
#         out = F.leaky_relu(out)
#         # ======================================
#
#         return self.gamma * out + input


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

# Squeeze and Excitation block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# SSPCAB implementation
class SSPCAB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1, reduction_ratio=8):
        '''
            channels: The number of filter at the output (usually the same with the number of filter from the input)
            kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
            dilation: The dilation dimension 'd' from the paper
            reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
        '''
        super(SSPCAB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2*dilation + 1

        self.relu = nn.ReLU()
        self.se = SELayer(channels, reduction_ratio=reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)

        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        x = self.relu(x1 + x2 + x3 + x4)

        x = self.se(x)
        return x


# Example of how our block should be updated
# mse_loss = nn.MSELoss()
# cost_sspcab = mse_loss(input_sspcab, output_sspcab)