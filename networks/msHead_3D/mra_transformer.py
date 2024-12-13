
import math
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)
model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)


import torch
import torch.nn as nn
import torch.nn.init as init
from mra_helper import Block, PatchEmbed
from ptflops import get_model_complexity_info
# import sys
# sys.path.append('..')
# from configs.config_imagenet import config as cfg

from timm.models.layers import trunc_normal_
from functools import partial
# from utils.logger import get_logger




# How to apply multihead multiscale
class MRATransformer(nn.Module):
    def __init__(self, img_size=(14, 224, 224), patch_size=(1, 4, 4), in_chans=1, num_classes=5, embed_dims=[48, 96, 192, 384], 
                 num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 2, 2]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        # self.logger = get_logger()
        self.img_size = img_size
        # print('img_size: ',img_size)

        # patch_embed
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=in_chans, 
                                       embed_dim=embed_dims[0], norm_layer=norm_layer)
        
        self.downsample_1 = PatchEmbed(img_size=(img_size[0]// 1, img_size[1]//4, img_size[2]//4),patch_size=(1, 2, 2), in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1], norm_layer=norm_layer)
        self.downsample_2 = PatchEmbed(img_size=(img_size[0]// 1, img_size[1]//8, img_size[2]//8), patch_size=(1, 2, 2), in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2], norm_layer=norm_layer)
        self.downsample_3 = PatchEmbed(img_size=(img_size[0]// 2, img_size[1]//16, img_size[2]//16), patch_size=(2, 2, 2), in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3], norm_layer=norm_layer)
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        # print(f'dpr: {dpr}')
        # 56x56
        
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, level = {-3: 1, -2:3, -1:3},
            img_size=(img_size[0]// 1, img_size[1]//4, img_size[2]//4))
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        # 28x28
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, level = {-3: 1, -2:2, -1:2},
            img_size=(img_size[0]//1, img_size[1]//8, img_size[2]//8))
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        # 14x14
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, level = {-3: 1, -2:1, -1:1},
            img_size=(img_size[0]//1, img_size[1]//16, img_size[2]//16))
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        #7x7
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, level = {},
            img_size=(img_size[0]//2, img_size[1]//32, img_size[2]//32))
            for i in range(depths[3])])             
        self.norm4 = norm_layer(embed_dims[3])

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(embed_dims[3], num_classes) if self.num_classes > 0 else nn.Identity()

        cur += depths[3]

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

    def forward_features(self, x_rgb):
        """
        x_rgb: B x C x D x H x W
        """
        print(f'input: {x_rgb.shape}')
        outs = []
        B, C, _, _, _ = x_rgb.shape
        stage = 0
        # B, N, C = B, Pd*Ph*Pw, C  --> Pd=(D//2), Ph=(H//2), Pw=(W//2)
        x_rgb, D, H, W = self.patch_embed(x_rgb)    # There is norm at the end of PatchEMbed
        print(f'x patch:{x_rgb.shape} D:{D} H:{H} W:{W}')

        # stage 1
        stage += 1        
        for j,blk in enumerate(self.block1):
            x_rgb = blk(x_rgb)
        # print('########### Stage 1 - Output: {}'.format(x_rgb.shape))
        x_out = self.norm1(x_rgb)
        x_out = x_out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x_out)
        x_rgb = x_rgb.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        x_rgb, D, H, W = self.downsample_1(x_rgb)       # There is norm at the end of PatchEMbed
        print(f'x_out stage 1:{x_rgb.shape} D:{D} H:{H} W:{W}')
        # stage 2
        stage += 1
        for j,blk in enumerate(self.block2):
            x_rgb = blk(x_rgb)
        x_out = self.norm2(x_rgb)
        x_out = x_out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x_out)
        x_rgb = x_rgb.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        x_rgb, D, H, W = self.downsample_2(x_rgb)       # There is norm at the end of PatchEMbed
        print(f'x_out stage 2:{x_rgb.shape} D:{D} H:{H} W:{W}')

        # stage 3
        stage += 1
        for j,blk in enumerate(self.block3):
            x_rgb = blk(x_rgb)
        x_out = self.norm3(x_rgb)
        x_out = x_out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x_out)
        x_rgb = x_rgb.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        x_rgb, D, H, W = self.downsample_3(x_rgb)       # There is norm at the end of PatchEMbed
        print(f'x_out stage 3:{x_rgb.shape} D:{D} H:{H} W:{W}')
        # stage 4
        stage += 1
        for j,blk in enumerate(self.block4):
            x_rgb = blk(x_rgb)
        x_rgb = self.norm4(x_rgb)
        x_rgb = x_rgb.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x_rgb)

        return outs

    def forward(self, x_rgb):
        # print()
        outs = self.forward_features(x_rgb)
        return outs

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
    def __init__(self, img_size, num_classes, embed_dims, depths, num_heads, drop_path_rate):
        super(mra_b0, self).__init__(
            img_size = img_size, patch_size = (1, 4, 4), num_classes=num_classes, embed_dims=embed_dims, 
            num_heads=num_heads, mlp_ratios=[4, 4, 4, 4], qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths, attn_drop_rate=0,
            drop_rate=0, drop_path_rate=drop_path_rate)




if __name__=="__main__":
    backbone = mra_b0(
        img_size=(14, 224, 224),
        num_classes=5,
        embed_dims=[48,96,192,384],
        depths=[2,2,2,2],
        num_heads = [3,6,12,24],
        drop_path_rate=0.1
    )
    
    # ########print(backbone)
    B = 2
    C = 1
    D = 14
    H = 224
    W = 224
    device = 'cuda:1'
    rgb = torch.randn(B, C, D, H, W)
    outputs = backbone(rgb)
    for i,out in enumerate(outputs):
        print(f'{i}:{out.size()}')

    total_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    macs, params = get_model_complexity_info(backbone, (1, 96, 96, 96), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))