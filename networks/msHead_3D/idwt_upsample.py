from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer


### Residual HF Refinement Block (Filters HF Before IDWT)
class HFRefinementRes(nn.Module):
    def __init__(self, in_channels, init_alpha=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True)
        self.norm = nn.InstanceNorm3d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # self.log_alpha = nn.Parameter(torch.log(torch.ones(in_channels) * init_alpha))

    def forward(self, x):
        refined = self.conv1(x)
        refined = self.norm(refined)
        refined = self.relu(refined)
        refined = self.conv2(refined)
        refined = self.sigmoid(refined)

        # alpha = F.softplus(self.log_alpha)  # shape: [C]
        # alpha_expanded = alpha.view(1, -1, 1, 1, 1)
        # out =  x * refined + alpha_expanded * x
        out = x * refined
        return out


class UnetrIDWTBlock(nn.Module):
    """
    Inverse Discrete Wavelet Transform (IDWT) Upsampling Block for UNETR.
    Uses HF refinement before IDWT to filter noise and enhance edges.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        stage: int,
        wavelet: str,
        kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            wavelet: wavelet type (e.g., 'db1', 'haar').
            kernel_size: convolution kernel size.
            norm_name: normalization type.
            res_block: if True, uses residual block instead of basic block.
        """
        super(UnetrIDWTBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet = wavelet

        # self.hf_refinement = []
        # for _ in range(stage):
        #     self.hf_refinement.append(HFRefinementRes(in_channels//pow(2, stage)))
        # self.hf_refinement = nn.ModuleList(self.hf_refinement)

        # Convolution for Low-Frequency (LF) components
        self.conv_lf_block = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
            is_transposed=False,
        )

        # Select residual or basic block
        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels * 2,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims,
                out_channels * 2,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip, hf_coeffs):
        """
        Forward pass.
        Args:
            inp: Low-frequency input from previous layer.
            skip: Skip connection from encoder.
            hf_coeffs: High-frequency coefficients from encoder.

        Returns:
            Refined and reconstructed feature map.
        """
        inp = self.conv_lf_block(inp)

        # **HF Refinement BEFORE IDWT**
        # hf_filtered = tuple(
        #     {key: self.hf_refinement[i](hf_dict[key]) for key in hf_dict} for i, hf_dict in enumerate(hf_coeffs)
        # )
        
        # Use raw hf_coeffs
        inp_tuple = (inp,) + hf_coeffs
        out = ptwt.waverec3(inp_tuple, wavelet=self.wavelet)  # IDWT Reconstruction

        # **Fuse reconstructed features with skip connection**
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)

        return out
