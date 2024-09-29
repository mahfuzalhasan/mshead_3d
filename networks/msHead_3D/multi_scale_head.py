import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from typing_extensions import Protocol
from typing import Sequence, Tuple, Union


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
import pywt
import ptwt


class WaveletTransform3D(torch.nn.Module):
    def __init__(self, wavelet='db1', level=5, mode='zero'):
        super(WaveletTransform3D, self).__init__()
        self.wavelet = wavelet #pywt.Wavelet(wavelet)
        self.level = level
        self.mode = mode

    def forward(self, x):
        # print(f'x:{x.shape}  ')
        coeffs = ptwt.wavedec3(x, wavelet=self.wavelet, level=self.level, mode=self.mode)
        Yl = coeffs[0]  # Extracting the approximation coefficients
        return Yl


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                    proj_drop=0., window_size=6, n_local_region_scale=3, dwt_layer_1=True, img_size=(48, 48, 48)):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.n_local_region_scale = n_local_region_scale
        self.window_size = window_size
        
        self.img_size = img_size
        self.D, self.H, self.W = img_size[0], img_size[1], img_size[2]
        self.N_G = self.D//self.window_size * self.H//self.window_size * self.W//self.window_size

        self.factor = int(self.H//self.window_size)
        self.level = int(math.log2(self.factor))

        self.dwt_layer_1 = dwt_layer_1

        self.dwt_downsamples = WaveletTransform3D(wavelet='haar', level=1)

        # Linear embedding
        self.qkv_proj = nn.Linear(dim, dim*3, bias=qkv_bias) 
        # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1) * (2 * self.window_size - 1),
                        self.num_heads))  

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w])) 
        coords_flatten = torch.flatten(coords, 1) 
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 2] += self.window_size - 1

        relative_coords[:, :, 0] *= 3 * self.window_size - 1
        relative_coords[:, :, 1] *= 2 * self.window_size - 1

        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

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

    
    # input: B, D, H, W, C
    # output: B*nWindow, window_size**3, C --> nWindow = number of windows = D//window_size(ws) * H//ws * W//ws
    def window_partition(self, arr):
        # print(arr.shape)
        B = arr.shape[0]
        D = arr.shape[1]
        H = arr.shape[2]
        W = arr.shape[3]
        C = arr.shape[4]
        
        arr_reshape = arr.view(B, D//self.window_size, self.window_size, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = arr_reshape.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, self.window_size, self.window_size, self.window_size, C)       
        return windows


    def attention(self, q, k, v):
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # scaling needs to be fixed
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size*self.window_size*self.window_size, self.window_size*self.window_size*self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        return x, attn
    

    def decomposition(self, x_local):
        x_local = x_local.permute(0, 4, 1, 2, 3).contiguous()   #B, C, D, H, W
        x_local = self.dwt_downsamples(x_local)
        x_local = x_local.permute(0, 2, 3, 4, 1).contiguous()   #B, D, H, W, C
        return x_local


    def forward(self, x):
        print('\n !!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        A = []
        D, H, W = self.D, self.H, self.W
        B, N, C = x.shape
        
        assert N==self.D*self.H*self.W
        
        attn_fused = 0
        x_local = x.view(B, D, H, W, C)
        # print(f'###################################')
        print(f'input:{x_local.shape}')
        for i in range(self.n_local_region_scale):
            up_required = False
            ############################# Wavelet Decomposition
            if self.dwt_layer_1 or i>0:
                x_local = self.decomposition(x_local)
                up_required = True

            print(f'branch: {i+1} x_local:{x_local.shape}')
            
            output_size = (x_local.shape[1], x_local.shape[2], x_local.shape[3])
            n_region = (output_size[0]//self.window_size) * (output_size[1]//self.window_size) * (output_size[2]//self.window_size)

            # print('x in attention after view: ',x.shape)
            x_windows = self.window_partition(x_local)
            x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)
            # print(f'windows:{x_windows.shape}')
            B_, Nr, C = x_windows.shape     # B_ = B * num_local_regions(num_windows), Nr = 6x6x6 = 216 (ws**3)
            
            ######## Attention
            qkv = self.qkv_proj(x_windows).reshape(B_, Nr, 3, C).permute(2, 0, 1, 3)   # temp--> 3, B_, Nr, C
            # 3, B*num_region_7x7, num_head, Nr, head_dim
            qkv = qkv.reshape(3, B_, Nr, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous() 
            q,k,v = qkv[0], qkv[1], qkv[2]      #B_, h, Nr, Ch
            #B_, h, Nr, Ch 
            y, attn = self.attention(q, k, v)
            #######################

            # B, num_head, Ch, num_region_6x6, Nr
            y = y.reshape(B, n_region, self.num_heads, Nr, self.head_dim).permute(0, 2, 4, 1, 3).contiguous()
            y = y.reshape(B, C, output_size[0], output_size[1], output_size[2])
            if up_required:
                y = F.interpolate(y, size=(self.D, self.H, self.W), mode='trilinear')

            y = y.view(B, self.num_heads, self.head_dim, self.D, self.H, self.W).permute(0, 1, 3, 4, 5, 2).contiguous()
            attn_fused += y
        
        attn_fused = attn_fused.reshape(B, self.num_heads, -1, C//self.num_heads)
        attn_fused = attn_fused.permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        attn_fused = self.proj(attn_fused)
        attn_fused = self.proj_drop(attn_fused)
        return attn_fused

    def flops(self):
        # FLOPs for linear layers
        flops_linear_q = 2 * self.dim * self.dim
        flops_linear_kv = 2 * self.dim * self.dim * 2
        head_dim = self.dim // self.num_heads
        flops = 0
        # print("number of heads ", self.num_heads)
        for i in range(self.num_heads):
            r_size = self.local_region_shape[i]
            if r_size == 1:
                N = self.H * self.W
                flops_attention_weight = N * head_dim * N
                flops_attention_output = N * N * head_dim

            else:
                region_number = (self.H * self.W) // (r_size ** 2)
                p = r_size ** 2
                flops_attention_weight = region_number * (p * head_dim * p)
                flops_attention_output = region_number * (p * p * head_dim)
            flops_head = flops_attention_weight + flops_attention_output
            flops += flops_head    

        total_flops = flops_linear_q + flops_linear_kv + flops
        return total_flops


        
if __name__=="__main__":
    # #######print(backbone)
    B = 4
    C = 128
    H = 56
    W = 56
    # device = 'cuda:1'
    ms_attention = MultiScaleAttention(C, num_heads=4, n_local_region_scale=4, window_size=7, img_size=(56, 56))
    # ms_attention = ms_attention.to(device)
    # # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, H*W, C)
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, H, W)
    print('output: ',y.shape)