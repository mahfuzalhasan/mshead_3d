
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

class UpsampleBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        wavelet:str,
        kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """
        
        super(UpsampleBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet = wavelet
        # self.transp_conv = get_conv_layer(
        #     spatial_dims,
        #     in_channels,
        #     out_channels,
        #     kernel_size=upsample_kernel_size,
        #     stride=upsample_stride,
        #     conv_only=True,
        #     is_transposed=True,
        # )
        self.conv_lf_block = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
            is_transposed=False,
        )
        # self.conv_lf_block = UnetBasicBlock(  # type: ignore
        #         spatial_dims,
        #         in_channels,
        #         out_channels,
        #         kernel_size=kernel_size,
        #         stride=1,
        #         norm_name=norm_name,
        # )
        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        # print(f'input: {inp.shape} skip:{skip.shape} in:{self.in_channels} out:{self.out_channels}')
        inp = self.conv_lf_block(inp)
        out = F.interpolate(inp, size=(skip.shape[2:]), mode='trilinear')
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out