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

class psa_fc_block_layer_2(nn.Module):
    # 分组卷积(conv_group)相当于稀疏卷积，意义就是减少参数量，多尺度卷积核会导致参数量上升，group操作可以减少参数
    # group这个操作在alexnet就有 conv_groups=[1, 4, 8, 16]
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(psa_fc_block_layer_2, self).__init__()
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
        x1_msa = self.att_MSA(x1)
        x2_msa = self.att_MSA(x2)
        # x3_msa = self.att_MSA(x3)
        # x4_msa = self.att_MSA(x4)
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


class psa_fc_block_layer_4(nn.Module):
    # 分组卷积(conv_group)相当于稀疏卷积，意义就是减少参数量，多尺度卷积核会导致参数量上升，group操作可以减少参数
    # group这个操作在alexnet就有 conv_groups=[1, 4, 8, 16]
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(psa_fc_block_layer_4, self).__init__()
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
        x1_msa = self.att_MSA(x1)
        x2_msa = self.att_MSA(x2)
        x3_msa = self.att_MSA(x3)
        x4_msa = self.att_MSA(x4)
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