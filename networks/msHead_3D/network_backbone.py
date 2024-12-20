#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:04:06 2022

@author: leeh43
"""

from typing import Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)
model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)

import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from ptflops import get_model_complexity_info

from monai.networks.nets import UNETR, SwinUNETR
from monai.networks.blocks.dynunet_block import UnetOutBlock, get_conv_layer
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from typing import Union

from lib.models.tools.module_helper import ModuleHelper
# from networks.UXNet_3D.uxnet_encoder import uxnet_conv
from networks.msHead_3D.mra_transformer import mra_b0
from idwt_upsample import UnetrIDWTBlock



sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        # Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)
    

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelCalibration(nn.Module):
    def __init__(self, in_channels=384, reduction_ratio=4, norm_layer=nn.BatchNorm3d):
        """
        SENet-style bottleneck with normalization.
        Args:
            in_channels: Number of input channels.
            reduction_ratio: Reduction ratio for channel squeeze.
            norm_layer: Normalization layer to use (e.g., BatchNorm3d or InstanceNorm3d).
        """
        super(ChannelCalibration, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        # Dimensionality Reduction and Expansion
        self.reduce = nn.Conv3d(in_channels, reduced_channels, kernel_size=1)
        self.norm_reduce = norm_layer(reduced_channels)

        self.conv = nn.Conv3d(reduced_channels, reduced_channels, kernel_size=3, padding=1)
        self.norm_conv = norm_layer(reduced_channels)

        self.expand = nn.Conv3d(reduced_channels, in_channels, kernel_size=1)
        self.norm_expand = norm_layer(in_channels)

        # Squeeze-and-Excitation (Global Attention)
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # Global average pooling
        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, in_channels)

        # Residual connection
        self.residual = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, in_channels, D, H, W).
        Returns:
            Output tensor of shape (B, in_channels, D, H, W).
        """
        identity = self.residual(x)  # Residual connection

        # Dimensionality reduction and spatial refinement
        x = self.reduce(x)
        x = self.norm_reduce(x)  # Normalize reduced features
        x = self.relu(x)

        x = self.conv(x)
        x = self.norm_conv(x)  # Normalize after spatial convolution
        x = self.relu(x)

        x = self.expand(x)
        x = self.norm_expand(x)  # Normalize expanded features

        # Squeeze-and-Excitation
        b, c, _, _, _ = x.shape
        se = self.global_pool(x).view(b, c)  # Global pooling to (B, C)
        se = F.relu(self.fc1(se))  # First FC layer
        se = self.sigmoid(self.fc2(se))  # Second FC layer + sigmoid
        se = se.view(b, c, 1, 1, 1)  # Reshape to (B, C, 1, 1, 1)
        x = x * se  # Scale channels

        # Add residual and return
        return self.relu(x + identity)





class MSHEAD_ATTN(nn.Module):

    def __init__(
        self,
        img_size = (96,96,96),
        patch_size = 2,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        num_heads = [3, 6, 12, 24],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
        use_checkpoint = False
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        # self.feature_size = feature_size
        self.num_heads = num_heads
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.depths)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        self.multiscale_transformer = mra_b0(
            img_size = self.img_size,
            patch_size= self.patch_size,
            num_classes = out_chans,
            embed_dims = self.feat_size,
            depths=self.depths,
            num_heads = self.num_heads,
            drop_path_rate=self.drop_path_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder10 = ChannelCalibration(
                in_channels=self.feat_size[3], 
                reduction_ratio=4,
                norm_layer=nn.InstanceNorm3d
        )

        # self.encoder10 = UnetrBasicBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[3],
        #     kernel_size=1,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )

        self.decoder4 = UnetrIDWTBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            wavelet='db1',
            kernel_size=3,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder3 = UnetrIDWTBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[1],
            wavelet='db1',
            kernel_size=3,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2 = UnetrIDWTBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[0],
            wavelet='db1',
            kernel_size=3,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.learnable_up4 = nn.ConvTranspose3d(self.feat_size[2], self.feat_size[0], kernel_size=4, stride=4)
        self.learnable_up3 = nn.ConvTranspose3d(self.feat_size[1], self.feat_size[0], kernel_size=2, stride=2)
        
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2]+self.feat_size[1]+self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        outs, outs_hf = self.multiscale_transformer(x_in)
        
        print(f'output from ms transformer: \n')
        for i,out in enumerate(outs):
            print(f'{i}:{out.shape}')
        
        # for i,hfs in enumerate(outs_hf):
        #     print(f'layer {i} hfs')
        #     for coeff in hfs:
        #         print(f'type {type(coeff)}')
        #         for k,cf in coeff.items():
        #             print(f'key: {k} - {cf.shape}')
        #             # print(type(cf))

        enc0 = self.encoder1(x_in)
        print(f'enc0 input:{x_in.shape} output:{enc0.size()}')

        enc1 = self.encoder2(outs[0])
        print(f'enc1 input:{outs[0].shape} output:{enc1.size()}')

        enc2 = self.encoder3(outs[1])
        print(f'enc2:input:{outs[1].shape} output:{enc2.size()}')

        enc3 = self.encoder4(outs[2])
        print(f'enc3:input:{outs[2].shape} output:{enc3.size()}')

        dec5 = self.encoder10(outs[3])
        print(f'bottleneck:{dec5.shape}')

        dec4 = self.decoder4(dec5, enc3, outs_hf[-1])
        print(f'dec4: {dec4.shape}')
        dec3 = self.decoder3(dec5, enc2, outs_hf[-2])
        print(f'dec3: {dec3.shape}')
        dec2 = self.decoder2(dec5, enc1, outs_hf[-3])
        print(f'dec2: {dec2.shape}')

        # Upsample dec4 and dec3 to match dec2 resolution
        # dec4_upsampled = F.interpolate(dec4, size=dec2.shape[2:], mode="trilinear", align_corners=False)
        # dec3_upsampled = F.interpolate(dec3, size=dec2.shape[2:], mode="trilinear", align_corners=False)
        # Learnable upsampling
        dec4_upsampled = self.learnable_up4(dec4)
        dec3_upsampled = self.learnable_up3(dec3)
        print(f'upsampled dec4:{dec4_upsampled.shape} dec3:{dec3_upsampled.shape}')

        # Fuse all decoder features
        combined = torch.cat([dec4_upsampled, dec3_upsampled, dec2], dim=1)  # Concatenate along channel dimension
        print(f'combined shape:{combined.shape}')

        dec1 = self.decoder1(combined, enc0)
        print(f'dec1: {dec1.shape}')
        
        return self.out(dec1)
    
    
    
    

if __name__=="__main__":
    B = 2
    C = 1
    D = 96
    H = 96
    W = 96
    num_classes = 5
    img_size = (D,H,W)
    model = MSHEAD_ATTN(
        img_size=(D, H, W),
        patch_size=2,
        in_chans=1,
        out_chans=num_classes,
        depths=[2,2,2,2],
        feat_size=[48,96,192,384],
        num_heads = [3,6,12,24],
        drop_path_rate=0.1,
        use_checkpoint=False,
    )
    # model = SwinUNETR(
    #     img_size=(D, H, W),
    #     in_channels=1,
    #     out_channels=num_classes,
    #     feature_size=48,
    #     use_checkpoint=False,
    # )
    model.cuda()
    x = torch.randn(B, C, D, H, W).cuda()
    # # Hook to record input and output shapes
    # def hook_fn(module, input, output):
    #     print(f"{module.__class__.__name__}:")
    #     print(f"    Input Shape: {input[0].shape}")
    #     print(f"    Output Shape: {output[0].shape}")

    # # Register the hook for all layers
    # for layer in model.children():
    #     layer.register_forward_hook(hook_fn)
    outputs = model(x)
    print(f'outputs: {outputs.shape}')

    # # Assuming 'model' is your PyTorch model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    # summary(model, (2, 1, 96, 96, 96), verbose=2)
    

  
    # macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity ptflops: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters from ptflops: ', params))