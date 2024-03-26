import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from skimage.measure import label, regionprops
import pdb

from .axial_atten_3d import AA_kernel
# from cross_attention import CrossAttentionBlock
from .self_attention import SelfAttentionBlock
from .conv_layer import Conv, Conv3D


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    #print(net)
    #print('Total number of parameters: %d' % num_params)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
                nn.Conv3d(channel, channel//reduction, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, kernel_size=1, stride=1),
                nn.Sigmoid()
                )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    if kernel_size == (1,3,3):
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                padding=(0,1,1), bias=False, dilation=dilation_rate)

    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
                padding=padding, bias=False, dilation=dilation_rate)

def conv2x2(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
            padding=padding, bias=False, dilation=dilation_rate)

class heatmap_pred(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(heatmap_pred, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_ch)
        self.conv1 = conv3x3(in_ch, in_ch)

        self.bn2  = nn.BatchNorm3d(in_ch)
        self.conv2 = conv3x3(in_ch, in_ch)

        self.conv3  = nn.Conv3d(in_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(out)

        out = self.sigmoid(out)

        return out
    

class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, reduction=4, dilation_rate=1, norm='bn'):
        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, kernel_size=kernel_size, stride=stride)
        if norm == 'bn':
            self.bn1 = nn.BatchNorm3d(inplanes)
        elif norm =='in':
            self.bn1 = nn.InstanceNorm3d(inplanes)
        elif norm =='gn':
            self.bn1 = nn.GroupNorm(NUM_GROUP, inplanes)
        else:
            raise ValueError('unsupport norm method')
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=dilation_rate)
        if norm == 'bn':
            self.bn2 = nn.BatchNorm3d(planes)
        elif norm =='in':
            self.bn2 = nn.InstanceNorm3d(planes)
        elif norm =='gn':
            self.bn2 = nn.GroupNorm(NUM_GROUP, planes)
        else:
            raise ValueError('unsupport norm method')
        self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if norm == 'bn':
                self.shortcut = nn.Sequential(
                    nn.BatchNorm3d(inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            elif norm =='in':
                self.shortcut = nn.Sequential(
                    nn.InstanceNorm3d(inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            elif norm =='gn':
                self.shortcut = nn.Sequential(
                    nn.GroupNorm(NUM_GROUP, inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                )
            else:
                raise ValueError('unsupport norm method')

        self.stride = stride

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.se(out)

        out += self.shortcut(residue)

        return out

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, norm='bn'):
        super(inconv, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = SEBasicBlock(out_ch, out_ch, kernel_size=(1,3,3), norm=norm)

    def forward(self, x): 

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        return out 

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se=False, reduction=2, dilation_rate=1, norm='bn'):
        super(conv_block, self).__init__()

        self.conv = SEBasicBlock(in_ch, out_ch, stride=stride, reduction=reduction, dilation_rate=dilation_rate, norm=norm)

    def forward(self, x):

        out = self.conv(x)

        return out
class ReverseAxialAttention(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes):
        super(ReverseAxialAttention, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_classes = num_classes
        # self.out_conv = nn.Conv2d(in_ch, 1, 1, 1)
        self.out_conv = nn.Sequential(
                            nn.Conv3d(in_ch, in_ch//2, 3, 1, 1),
                            nn.Conv3d(in_ch//2, self.num_classes, 1, 1))
        self.aa_kernel = AA_kernel(out_ch, out_ch)

        self.ra_conv1 = Conv3D(out_ch,out_ch,3,1,padding=1,bn_acti=True)
        self.ra_conv2 = Conv3D(out_ch,out_ch,3,1,padding=1,bn_acti=True)
        self.ra_conv3 = Conv3D(out_ch,self.num_classes,3,1,padding=1,bn_acti=True)

    def forward(self, dec_out, enc_out):
        partial_output = self.out_conv(dec_out)
        if self.num_classes==1:
            partial_output_ra = -1*(torch.sigmoid(partial_output)) + 1
        aa_attn = self.aa_kernel(enc_out)
        #print(f'aa attn:{aa_attn.shape} partial_out:{partial_output_ra.shape}')
        aa_attn_o = partial_output_ra.expand(-1, self.out_ch, -1, -1, -1).mul(aa_attn)
        #print(f'aa_attn_o:{aa_attn_o.shape}')

        ra =  self.ra_conv1(aa_attn_o) 
        ra = self.ra_conv2(ra) 
        ra = self.ra_conv3(ra)

        out = ra + partial_output

        return out


class up_block_cross_attn(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2, 2), se=False, reduction=2, norm='bn'):
        super(up_block_cross_attn, self).__init__()

        self.scale = scale

        self.conv = nn.Sequential(
            conv_block(in_ch+out_ch, out_ch, se=se, reduction=reduction, norm=norm),
        )
        # self.ra_attn = ReverseAxialAttention(in_ch+out_ch, out_ch)

        # self.ca = CrossAttentionBlock(in_channels=out_ch, key_channels=out_ch//2, value_channels=out_ch//4 )
        self.attn_block = SelfAttentionBlock(in_channels=in_ch+out_ch, key_channels=out_ch, value_channels=out_ch//2 )

    def sparse_attention(self, x):
        N, C, H, W = x.shape
        p_h = p_w = 12
        q_h = H // p_h
        q_w = W // p_w
        x_r = x.reshape(N, C, q_h, p_h, q_w, p_w)
        x_p = x_r.permute(0, 3, 5, 1, 2, 4)
        x = x_p.reshape(N * p_h * p_w, C, q_h, q_w)
       
        global_relation = self.attn_block(x)
        
        gr_r = global_relation.reshape(N, p_h, p_w, C, q_h, q_w)
        gr_p = gr_r.permute(0, 4, 5, 3, 1, 2)
        gr = gr_p.reshape(N * q_h * q_w, C, p_h, p_w)
        attn_out = self.attn_block(gr)

        x = attn_out.reshape(N, q_h, q_w, C, p_h, p_w)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)
        return x

    def forward(self, x_dec, x_enc):  #x1 from dec and x2 fro encoder
        x_dec = F.interpolate(x_dec, scale_factor=self.scale, mode='nearest')
        out = torch.cat([x_enc, x_dec], dim=1)
        out = self.sparse_attention(out)
        out = self.conv(out)
        return out


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes=1, scale=(2, 2, 2), se=False, reduction=2, norm='bn'):
        super(up_block, self).__init__()

        self.scale = scale

        self.conv = nn.Sequential(
            conv_block(in_ch+out_ch, out_ch, se=se, reduction=reduction, norm=norm)
        )
        # self.ra_attn = ReverseAxialAttention(in_ch+out_ch, out_ch, num_classes=num_classes)

        # self.ra_attn = ReverseAxialAttention(in_ch, out_ch, num_classes=num_classes)

    def forward(self, x_dec, x_enc):  #x1 from dec and x2 fro encoder
        x_dec = F.interpolate(x_dec, scale_factor=self.scale, mode='nearest')
        #print(f'x_enc:{x_enc.shape} x_dec:{x_dec.shape}')
        out = torch.cat([x_enc, x_dec], dim=1)
        #print(f'concat:{out.shape}')
        # ra_out = self.ra_attn(out, x_enc)    #with concatenated feature
        # print(f'ra out:{ra_out.shape}')
        # ra_out = self.ra_attn(x_dec, x_enc)      #with only decoder feature
        out = self.conv(out)
        return out

class up_nocat(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2,2,2), se=False, reduction=2, norm='bn'):
        super(up_nocat, self).__init__()

        self.scale = scale
        self.conv = nn.Sequential(
            conv_block(out_ch, out_ch, se=se, reduction=reduction, norm=norm),
        )

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=True)
        out = self.conv(out)

        return out

class literal_conv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, reduction=2, norm='bn'):
        super(literal_conv, self).__init__()
        self.conv = conv_block(in_ch, out_ch, se=se, reduction=reduction, norm=norm)
    def forward(self, x):
        out = self.conv(x)
        return out

class DenseASPPBlock(nn.Sequential):
    """Conv Net block for building DenseASPP"""

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm='bn'):
        super(DenseASPPBlock, self).__init__()
        if bn_start:
            if norm == 'bn':
                self.add_module('norm_1', nn.BatchNorm3d(input_num))
            elif norm == 'in':
                self.add_module('norm_1', nn.InstanceNorm3d(input_num))
            elif norm == 'gn':
                self.add_module('norm_1', nn.GroupNorm(NUM_GROUP, input_num))

        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1', nn.Conv3d(in_channels=input_num, out_channels=num1, kernel_size=1))

        if norm == 'bn':
            self.add_module('norm_2', nn.BatchNorm3d(num1))
        elif norm == 'in':
            self.add_module('norm_2', nn.InstanceNorm3d(num1))
        elif norm == 'gn':
            self.add_module('norm_2', nn.GroupNorm(NUM_GROUP, num1))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2', nn.Conv3d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate))

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseASPPBlock, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)

        return feature