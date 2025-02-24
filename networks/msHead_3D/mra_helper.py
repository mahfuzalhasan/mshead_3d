import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import LayerNorm

import itertools
from functools import partial
from einops import rearrange
import os
import sys
import math
import time
import ptwt


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)
model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)

from multi_scale_head import WindowAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from configs.config_imagenet import config
# from utils.logger import get_logger



class ProjectionUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, residual=True, use_double_conv=False):
        super(ProjectionUpsample, self).__init__()

        self.do_res = residual
        self.stride = stride
        self.use_double_conv = use_double_conv

        # Bilinear Upsampling + 3x3x3 Depthwise Conv
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )

        # Channel-wise Interaction (1x1x1 conv)
        self.conv2 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=1, stride=1)

        # Channel Projection
        if self.use_double_conv:  # double conv for large reductions (e.g., 192 → 48)
            self.conv3 = nn.Sequential(
                nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
            )
        else:  # Apply single conv for small reductions (e.g., 96 → 48)
            self.conv3 = nn.Conv3d(in_channels * 2, out_channels, kernel_size=1)

        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)

        # Residual Path
        if self.do_res:
            self.res_conv = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
            )

        self.act = nn.GELU()

    def forward(self, x):
        x1 = x
        x1 = self.conv1(x1)  # Upsampling
        x1 = self.act(self.conv2(self.norm(x1)))  # Refinement
        x1 = self.conv3(x1)  # Final Projection

        if self.do_res:
            res = self.res_conv(x)  # Residual Connection
            x1 = x1 + res  # Merge Features

        return x1


# logger = get_logger()

class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.dim = dim

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous() # B N C -> B C N -> B C H W
        self.input_shape = x.shape
        x = self.dwconv(x) 
        x = x.flatten(2).transpose(1, 2) # B C H W -> B N C

        return x

    def flops(self):
        # Correct calculation for output dimensions
        padding = (1,1) 
        kernel_size = (3,3)
        stride = 1
        groups = self.dim
        in_chans = self.dim
        out_chans = self.dim

        output_height = ((self.input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride) + 1
        output_width = ((self.input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride) + 1

        # Convolution layer FLOPs
        conv_flops = 2 * out_chans * output_height * output_width * kernel_size[0] * kernel_size[1] * in_chans / groups

        total_flops = conv_flops
        return total_flops

class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class CCF_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0., img_size=(48, 48, 48)):
        super().__init__()
        """
        FFN Block
        """
        self.D, self.H, self.W = img_size[0], img_size[1], img_size[2]
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.C_hid = hidden_features
        self.pwconv = nn.Conv3d(in_features, hidden_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features)
        self.fc = nn.Linear(hidden_features, in_features)

        self.act = act_layer()
        self.norm1 = norm_layer(hidden_features)
        self.norm2 = norm_layer(hidden_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # Assuming trunc_normal_ is defined elsewhere
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            # Adapted for 3D convolution: including depth in the fan_out calculation
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, D, H, W, C = x.shape
        N = D * H * W
        assert N == self.D * self.H * self.W
        x_perm = x.permute(0, 4, 1, 2, 3).contiguous().view(B, C, D, H, W)
        
        p_out = self.pwconv(x_perm).reshape(B, self.C_hid, N).permute(0, 2, 1).contiguous()
        p_out = self.act(self.norm1(p_out))
        p_out = p_out.permute(0, 2, 1).reshape(B, self.C_hid, D, H, W)
        
        d_out = self.dwconv(p_out).reshape(B, self.C_hid, N).permute(0, 2, 1).contiguous()
        d_out = self.act(self.norm2(d_out))
        
        x_out = self.fc(d_out)
        x_out = x_out.view(B, D, H, W, -1)
        x = x + x_out
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        # x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def flops(self):
        # H, W = self.H, self.W
        flops_mlp = self.fc1.in_features * self.fc1.out_features * 2
        flops_mlp += self.dwconv.flops()
        flops_mlp += self.fc2.in_features * self.fc2.out_features * 2
        return flops_mlp
    
class WaveletTransform3D(torch.nn.Module):
    def __init__(self, wavelet='db1', mode='zero'):
        super(WaveletTransform3D, self).__init__()
        self.wavelet = wavelet #pywt.Wavelet(wavelet)
        self.mode = mode

    def forward(self, x, level):
        # print(f'x:{x.shape}  ')
        coeffs = ptwt.wavedec3(x, wavelet=self.wavelet, level=level, mode=self.mode)
        Yl  = coeffs[0]  # Extracting the approximation coefficients
        Yh = coeffs[1:]
        # print(f'Yl:{Yl.shape}')
        return Yl, Yh


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, level=0, img_size=(48, 48, 48)):
        super().__init__()
        self.dim = dim 
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.level = level
        mlp_hidden_dim = int(dim * mlp_ratio)

        if self.level > 0:
            self.dwt_downsamples = WaveletTransform3D(wavelet='db1')
        self.window_size = self.img_size[0]//pow(2, level)
        self.attn_computation_level = self.level if self.level > 0 else 1

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=self.window_size, img_size=img_size)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        self.mlp = CCF_FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, norm_layer=norm_layer, drop=drop, img_size=img_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, D, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, window_size C)
        """
        B, D, H, W, C = x.shape
        x = x.view(B,  D // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
        return windows

    def forward(self, x):
        D,H,W = self.img_size
        B,_,_,_,C = x.shape
        assert D == x.shape[1]
        assert H == x.shape[2]
        assert W == x.shape[3]
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        attn_fused = 0
        hfs = []

        for i in range(self.attn_computation_level):
            if self.level > 0:
                x = x.permute(0, 4, 1, 2, 3).contiguous()#B,C,D,H,W
                x, x_h = self.dwt_downsamples(x, 1)
                x = x.permute(0, 2, 3, 4, 1).contiguous() #B,D1,H1,W1,C
            # print(f'DWT_x:{x.shape} {x.dtype} shortcut:{shortcut.shape}')
            # print(f"type:{type(x_h)}")
            # for coeff in x_h:
            #     print(f'type {type(coeff)}')
            #     for k,cf in coeff.items():
            #         print(f'key: {k} - {cf.shape}- {cf.dtype}')
            output_size = (x.shape[1], x.shape[2], x.shape[3])
            nW = (output_size[0]//self.window_size) * (output_size[1]//self.window_size) * (output_size[2]//self.window_size)

            x_windows = self.window_partition(x, self.window_size)
            
            x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)
            B_, Nr, C = x_windows.shape     # B_ = B * num_local_regions(num_windows), Nr = 6x6x6 = 216 (ws**3)
            
            # B*nW, Nr, C
            attn_windows = self.attn(x_windows) 
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C).reshape(B, nW, self.window_size, self.window_size, self.window_size, C)   # B, D, H, W, C [Here nW = 1]
            attn_windows = attn_windows.reshape(B, output_size[0], output_size[1], output_size[2], C)
            attn_windows = attn_windows.permute(0, 4, 1, 2, 3).contiguous()         # B, C, D1, H1, W1
            if self.level > 0:
                # inp_tuple = (x,) + x_h
                # x = ptwt.waverec3(inp_tuple, wavelet='db1')
                attn_fused = attn_fused + F.interpolate(attn_windows, size=(D, H, W), mode='trilinear')   # B, C, D, H, W
                hfs.extend(x_h)
            else:
                attn_fused = attn_fused + attn_windows

        attn_fused = attn_fused.permute(0, 2, 3, 4, 1).contiguous()                    # B, D, H, W, C
        attn_fused = shortcut + self.drop_path(attn_fused)
        attn_fused = attn_fused + self.drop_path(self.mlp(self.norm2(attn_fused)))     # B, D, H, W, C
        if self.level > 0:
            return attn_fused, tuple(reversed(hfs))
        return attn_fused
    
    def flops(self):
        # FLOPs for MultiScaleAttention
        attn_flops = self.attn.flops()

        # FLOPs for Mlp
        mlp_flops = self.mlp.flops()

        total_flops = attn_flops + mlp_flops
        return total_flops


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.input_shape = None

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # B C H W
        self.input_shape = x.shape
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)
        return x, H, W

    def flops(self):
        # Correct calculation for output dimensions
        padding = (self.patch_size[0] // 2, self.patch_size[1] // 2)
        output_height = ((self.input_shape[2] + 2 * padding[0] - self.patch_size[0]) // self.stride) + 1
        output_width = ((self.input_shape[3] + 2 * padding[1] - self.patch_size[1]) // self.stride) + 1

        # Convolution layer FLOPs
        conv_flops = 2 * self.embed_dim * output_height * output_width * self.patch_size[0] * self.patch_size[1] * self.in_chans

        # Layer normalization FLOPs
        norm_flops = 2 * self.embed_dim * output_height * output_width

        total_flops = conv_flops + norm_flops
        return total_flops

    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 96, 96, 96
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        use_conv_embed (bool): Wherther use overlapped convolutional embedding layer. Default: False.
        norm_layer (nn.Module, optional): Normalization layer. Default: None 
        use_pre_norm (bool): Whether use pre-normalization before projection. Default: False
        is_stem (bool): Whether current patch embedding is stem. Default: False
    """

    def __init__(self, img_size=(96, 96, 96), patch_size=2, in_chans=1, embed_dim=48, 
                    use_conv_embed=False, norm_layer=None, use_pre_norm=False, is_stem=False):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size,  img_size[2] // patch_size]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm
        self.use_conv_embed = use_conv_embed

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.kernel_size = kernel_size
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(patch_size, patch_size, patch_size), stride=(patch_size, patch_size, patch_size))
            # self.conv1=nn.Conv3d(in_chans, embed_dim, kernel_size=3,stride=patch_size,padding=1)
            # self.conv2=nn.Conv3d(embed_dim, embed_dim,kernel_size=3,stride=1,padding=1)
        

        if self.use_pre_norm:
            if norm_layer is not None:
                self.pre_norm = nn.GroupNorm(1, in_chans)
            else:
                self.pre_norm = None

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    # input : B, C, D, H, W
    # output: B, N, C = B, Pd*Ph*Pw, C  --> Pd=(D//2),Ph=(H//2), Pw=(W//2)
    def forward(self, x):
        B, C, D, H, W = x.shape
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.use_pre_norm:
            x = self.pre_norm(x)

        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()  # B Pd*Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, D, H, W

    def flops(self):
        Ho, Wo = self.patches_resolution
        if self.use_conv_embed:
            flops = Ho * Wo * self.embed_dim * self.in_chans * (self.kernel_size**2)
        else:
            flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]