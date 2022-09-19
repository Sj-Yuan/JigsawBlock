# PFAM: Pretrained_Feature_Attention_Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block.MFCSAM import SELayer,mfcsa_module_layer_2,mfcsa_module_layer_4

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PFAM(nn.Module):

    def __init__(self, layer1_channel, layer3_channel, conv_kernels = [3, 5, 7, 9], conv_groups = [1, 4, 8, 16]):
        super().__init__()

        self.conv1_ = conv1x1(layer1_channel, layer1_channel)

       
        # MFCSAM
        self.att_mfcsam_2_x1 = mfcsa_module_layer_2(layer1_channel, layer1_channel, stride=1, conv_kernels=conv_kernels,
                                                    conv_groups=conv_groups)
        self.att_mfcsam_4_x1 = mfcsa_module_layer_4(layer1_channel, layer1_channel, stride=1, conv_kernels=conv_kernels,
                                                 conv_groups=conv_groups)
        self.att_mfcsam_2 = mfcsa_module_layer_2(layer3_channel, layer3_channel, stride=1, conv_kernels=conv_kernels,
                                                 conv_groups=conv_groups)
        self.att_mfcsam_4 = mfcsa_module_layer_4(layer3_channel, layer3_channel, stride=1, conv_kernels=conv_kernels,
                                                 conv_groups=conv_groups)

    def forward(self, input):
        x_1 = input[0]
        batch_size_1, channels_1, height_1, width_1 = x_1.shape
        x_2 = input[1]
        x_3 = input[2]
        
        batch_size_3, channels_3, height_3, width_3 = x_3.shape

        x_1_atten = self.att_mfcsam_2_x1(x_1)
        x_1_conv = self.conv1_(x_1)
        x_1 = x_1_conv + x_1_atten

        x_3 = self.att_mfcsam_2(x_3)
        
        output = [x_1, x_2, x_3]
        return output
