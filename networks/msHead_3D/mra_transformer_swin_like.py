
import math
import time
import sys
import os
from functools import partial
from ptflops import get_model_complexity_info

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)
model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.layers import trunc_normal_

from monai.networks.blocks import PatchEmbed
from monai.utils import  optional_import
rearrange, _ = optional_import("einops", name="rearrange")

from mra_helper import Block, PatchMerging
# from utils.logger import get_logger




# How to apply multihead multiscale
class MRATransformer(nn.Module):
    def __init__(self, img_size=(96, 96, 96), patch_size=2, in_chans=1, num_classes=5, embed_dims=[48, 96, 192, 384], 
                 num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., spatial_dims=3, norm_layer=nn.LayerNorm, patch_norm=False, 
                 depths=[2, 2, 2, 2]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.patch_norm = patch_norm
        self.patch_size = patch_size
        self.img_size = img_size
        print('img_size: ',img_size)

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # # patch_embed
        
        # self.patch_embed2 = PatchEmbed(img_size=(img_size[0]// 2, img_size[1]//2, img_size[2]//2),patch_size=2, in_chans=embed_dims[0],
        #                                       embed_dim=embed_dims[1], norm_layer=norm_layer)
        # self.patch_embed3 = PatchEmbed(img_size=(img_size[0]// 4, img_size[1]//4, img_size[2]//4), patch_size=2, in_chans=embed_dims[1],
        #                                       embed_dim=embed_dims[2], norm_layer=norm_layer)
        # self.patch_embed4 = PatchEmbed(img_size=(img_size[0]// 8, img_size[1]//8, img_size[2]//8), patch_size=2, in_chans=embed_dims[2],
        #                                       embed_dim=embed_dims[3], norm_layer=norm_layer)
        # # self.patch_embed5 = PatchEmbed(img_size=(img_size[0]// 16, img_size[1]//16, img_size[2]//16), patch_size=2, in_chans=embed_dims[3],
        # #                                       embed_dim=embed_dims[4], norm_layer=norm_layer)
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        # print(f'dpr: {dpr}')
        
        
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, level = 3,
            img_size=(img_size[0]// 2, img_size[1]//2, img_size[2]//2))
            for i in range(depths[0])])
        self.downsample_1 = PatchMerging(dim = embed_dims[0], norm_layer=norm_layer, spatial_dims=len(img_size))
        # self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, level = 2,
            img_size=(img_size[0]//4, img_size[1]//4, img_size[2]//4))
            for i in range(depths[1])])
        self.downsample_2 = PatchMerging(dim = embed_dims[1], norm_layer=norm_layer, spatial_dims=len(img_size))
        # self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, level = 1,
            img_size=(img_size[0]//8, img_size[1]//8, img_size[2]//8))
            for i in range(depths[2])])
        self.downsample_3 = PatchMerging(dim = embed_dims[2], norm_layer=norm_layer, spatial_dims=len(img_size))
        # self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, level = 0,
            img_size=(img_size[0]//16, img_size[1]//16, img_size[2]//16))
            for i in range(depths[3])])
        self.downsample_4 = PatchMerging(dim = embed_dims[3], norm_layer=norm_layer, spatial_dims=len(img_size))             
        # self.norm4 = norm_layer(embed_dims[3])
        cur += depths[3]

        self.apply(self._init_weights)

    # x --> B, C, D, H, W
    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.shape
            # Force trace() to generate a constant by casting to int
            ch = int(x_shape[1])
            if len(x_shape) == 5:
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

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

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            self.load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')
    
    def load_dualpath_model(self, model, model_file):
    # load raw state_dict
        t_start = time.time()
        if isinstance(model_file, str):
            raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            #raw_state_dict = torch.load(model_file)
            if 'model' in raw_state_dict.keys():
                raw_state_dict = raw_state_dict['model']
        else:
            raw_state_dict = model_file
        

        t_ioend = time.time()

        model.load_state_dict(raw_state_dict, strict=False)
        #del state_dict
        t_end = time.time()
        self.logger.info("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(t_ioend - t_start, t_end - t_ioend))

    def forward_features(self, x_rgb, normalize = True):
        """
        x_rgb: B x C x D x H x W
        """
        print(f'input: {x_rgb.shape}')
        outs = []
        outs_hf = []
        B, C, D, H, W = x_rgb.shape
        ######## Patch Embedding
        x0 = self.patch_embed(x_rgb)                # B, c, d, h, w         
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)       # B, c, d, h, w
        outs.append(x0_out)
        ########################

        # stage 1
        x1 = rearrange(x0, "b c d h w -> b d h w c")
        b,d,h,w,c = x1.shape        
        for j,blk in enumerate(self.block1):
            x1, x_h = blk(x1)       # B, d, h, w, c
        # print('########### Stage 1 - Output: {}'.format(x_rgb.shape))
        x1 = self.downsample_1(x1)
        x1 = rearrange(x1, "b d h w c -> b c d h w")
        x1_out = self.proj_out(x1, normalize)
        outs.append(x1_out)
        outs_hf.append(x_h)
        #######################
        
        # stage 2
        x2 = rearrange(x1, "b c d h w -> b d h w c")
        b,d,h,w,c = x2.shape
        for j,blk in enumerate(self.block2):
            x2, x_h = blk(x2)
        x2 = self.downsample_2(x2)
        x2 = rearrange(x2, "b d h w c -> b c d h w")
        x2_out = self.proj_out(x2, normalize)
        outs.append(x2_out)
        outs_hf.append(x_h)
        #######################
        

        # stage 3
        x3 = rearrange(x2, "b c d h w -> b d h w c")
        b,d,h,w,c = x3.shape
        for j,blk in enumerate(self.block3):
            x3, x_h = blk(x3)
        x3 = self.downsample_3(x3)
        x3 = rearrange(x3, "b d h w c -> b c d h w")
        x3_out = self.proj_out(x3, normalize)
        outs.append(x3_out)
        outs_hf.append(x_h)
        ########################

        # stage 4
        x4 = rearrange(x3, "b c d h w -> b d h w c")
        b,d,h,w,c = x4.shape
        for j,blk in enumerate(self.block4):
            x4 = blk(x4)
        x4 = self.downsample_4(x4)
        x4 = rearrange(x4, "b d h w c -> b c d h w")
        x4_out = self.proj_out(x4, normalize)
        outs.append(x4_out)
        ########################

        return outs, outs_hf

    def forward(self, x_rgb):
        outs, outs_hf = self.forward_features(x_rgb)
        return outs, outs_hf

    def flops(self):
        flops = 0
        flops += self.patch_embed1.flops()
        flops += self.patch_embed2.flops()
        flops += self.patch_embed3.flops()
        flops += self.patch_embed4.flops()

        for i, blk in enumerate(self.block1):
            flops += blk.flops()
        for i, blk in enumerate(self.block2):
            flops += blk.flops()
        for i, blk in enumerate(self.block3):
            flops += blk.flops()
        for i, blk in enumerate(self.block4):
            flops += blk.flops()
        return flops


class mra_b0(MRATransformer):
    def __init__(self, img_size, patch_size, num_classes, embed_dims, depths, num_heads, drop_path_rate):
        super(mra_b0, self).__init__(
            img_size = img_size, patch_size = patch_size, num_classes=num_classes, embed_dims=embed_dims, 
            num_heads=num_heads, mlp_ratios=[4, 4, 4, 4], qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths, attn_drop_rate=0,
            drop_rate=0, drop_path_rate=drop_path_rate)




if __name__=="__main__":
    backbone = mra_b0(
        img_size=(96, 96, 96),
        patch_size=2,
        num_classes=5,
        embed_dims=[48,96,192,384],
        depths=[2,2,2,2],
        num_heads = [3,6,12,24],
        drop_path_rate=0.1
    )
    
    # ########print(backbone)
    B = 2
    C = 1
    D = 96
    H = 96
    W = 96
    device = 'cuda:1'
    rgb = torch.randn(B, C, D, H, W)
    outputs, outputs_hf = backbone(rgb)
    for i,out in enumerate(outputs):
        print(f'{i}:{out.size()}')

    total_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # macs, params = get_model_complexity_info(backbone, (1, 96, 96, 96), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))