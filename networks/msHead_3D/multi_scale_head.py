import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time

class MergeRegions(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads=3):
        super(MergeRegions, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel * num_heads, 
                              out_channels=out_channel * num_heads, 
                              kernel_size=2, stride=2, groups=num_heads)
    
    # B, num_local_head, num_regions_7x7, 49(7x7 flattened), head_dim 
    def forward(self, x):
        B, h, R, Nr, C_h = x.shape
        R_out = R//4
        r = int(math.sqrt(R))
       
        x = x.view(B, h * C_h, R, Nr)
        x_reshaped = x.view(B, h * C_h, r, r * Nr)
        x_merged = self.conv(x_reshaped)
        
        out_shape = (B, h, R_out, Nr, C_h)
        x_out = x_merged.view(out_shape)
        return x_out

class MultiScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                    proj_drop=0., n_local_region_scales = 3, window_size=6, img_size=(32, 32)):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        self.window_size = window_size
        
        self.n_local_region_scales = n_local_region_scales
        self.local_dim = self.dim//self.n_local_region_scales
        self.local_head = self.num_heads//self.n_local_region_scales
        
        self.img_size = img_size
        self.D, self.H, self.W = img_size[0], img_size[1], img_size[2]
        self.N_G = self.D//self.window_size * self.H//self.window_size * self.W//self.window_size

        assert self.num_heads%n_local_region_scales == 0
        # Linear embedding
        self.qkv_proj = nn.Linear(dim, dim*3, bias=qkv_bias) 
        # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1) * (2 * self.window_size - 1),
                        self.local_head))  

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
    

    def upsample(self, x):
        B, l_h, R, Nr, C_h = x.shape
        r = math.ceil(R**(1./3.))
        output_size = math.ceil(self.N_G**(1./3.))
        perm_x = x.permute(0, 1, 4, 3, 2).contiguous().reshape(B, l_h*C_h*Nr, r, r, r)
        upsampled_x = F.interpolate(perm_x, size=(output_size, output_size, output_size), mode='trilinear')
        upsampled_x = upsampled_x.reshape(B, l_h, C_h, Nr, self.N_G).permute(0, 1, 4, 3, 2).contiguous()
        return upsampled_x
    
    def merge_regions_spatial(self, x, merge_size):
        # x shape is expected to be (B, H, R, N, C_h)
        
        _, B, H, R, N, C_h = x.shape
        x = x.reshape(-1, H, R, N, C_h)
        B_, H, R, N, C_h = x.shape      # B_ = B*3, H=#num_head, R = #region_6x6x6=512, N = 216, C_h = self.head_dim
        # print(f'x:{x.shape}')
        
        # Determine the new grid size based on the merge size
        grid_size = math.ceil((R // (merge_size ** 3)) ** (1/3))
        # print(f'grid size:{grid_size}')
        new_R = grid_size ** 3  # Number of regions after merge
        
        r = math.ceil(R**(1/3))
        # print(f'r:{r}')
        x_reshaped = x.view(B_, H, r, r, r, N, C_h)
        
        # Apply pooling over the spatial dimensions representing the regions
        # while preserving the separate channel information
        x_avg = F.avg_pool3d(x_reshaped.permute(0, 1, 6, 5, 2, 3, 4).reshape(B_, H * C_h * N, r, r, r),
                                kernel_size=(merge_size, merge_size, merge_size),
                                stride=(merge_size, merge_size, merge_size)).reshape(B_, H, C_h, N, grid_size, grid_size, grid_size)
        
        x_max = F.max_pool3d(x_reshaped.permute(0, 1, 6, 5, 2, 3, 4).reshape(B_, H * C_h * N, r, r, r),
                                kernel_size=(merge_size, merge_size, merge_size),
                                stride=(merge_size, merge_size, merge_size)).reshape(B_, H, C_h, N, grid_size, grid_size, grid_size)
        
        # Reshape back to match the expected output format
        x_avg = x_avg.permute(0, 1, 4, 5, 6, 3, 2).contiguous().reshape(B_, H, new_R, N, C_h)
        x_max = x_max.permute(0, 1, 4, 5, 6, 3, 2).contiguous().reshape(B_, H, new_R, N, C_h)
        
        return (x_avg + x_max)



    def forward(self, x, D, H, W):
        #####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        # N = H*W
        self.D=D
        self.H=H
        self.W=W
        A = []
        B, N, C = x.shape
        # print('reshape: ',x.shape)
        assert N==self.D*self.H*self.W
        
        x = x.view(B, D, H, W, C)
        x_windows = self.window_partition(x)
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)
        B_, Nr, C = x_windows.shape     # B_ = B * num_local_regions(num_windows), Nr = 6x6x6 = 216 (ws**3)
        temp = self.qkv_proj(x).reshape(B_, Nr, 3, C).permute(2, 0, 1, 3)   # temp--> 3, B_, Nr, C
        # print(f'temp shape:{temp.shape}')

        self.attn_outcome_per_group = []
        self.attn_mat_per_head = []
        
        for i in range(self.n_local_region_scales):
            # print(f'################ {i} #####################')
            local_C = C//self.n_local_region_scales
            qkv = temp[:, :, :, i*local_C:i*local_C + local_C]
            # print(f'qkv shape: {qkv.shape}')
            # 3, B*num_region_6x6, num_local_head, Nr, head_dim
            qkv = qkv.reshape(3, B_, Nr, self.local_head, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
            # print(f'qkv mh:{qkv.shape} self.N_G:{self.N_G}')
            # exit()
            if i>0:
                #3, B, num_local_head, num_region_6x6x6, 216, head_dim 
                qkv = qkv.view(3, B, self.N_G, self.local_head, Nr, self.head_dim).permute(0, 1, 3, 2, 4, 5).contiguous()
                # print(f'qkv reshape: {qkv.shape}')
                #B*3, num_local_head, num_region_6x6x6, 216, head_dim
                qkv = self.merge_regions_spatial(qkv, merge_size=int(math.pow(2,i)))
                # print(f'qkv merged: {qkv.shape}')
                # 3, B_, num_local_head, Nr, head_dim
                qkv = qkv.permute(0, 2, 1, 3, 4).contiguous().reshape(3, -1, self.local_head, Nr, self.head_dim)

            q,k,v = qkv[0], qkv[1], qkv[2]      #B_, num_local_head, Nr, Ch
            # print(f'q:{q.shape} k:{k.shape} v:{v.shape}')
            
            y, attn = self.attention(q, k, v)
            # print(f'y:{y.shape} attn:{attn.shape}')
            
            output_size = (self.D//pow(2,i), self.H//pow(2,i), self.W//pow(2,i))
            n_region = (output_size[0]//self.window_size) * (output_size[1]//self.window_size) * (output_size[2]//self.window_size)
            # print(f'output size:{output_size} n_region:{n_region}')
            
            y = y.reshape(B, n_region, self.local_head, self.window_size* self.window_size * self.window_size, self.head_dim).permute(0, 2, 1, 3, 4)
            # print(f'y reshape:{y.shape}')
            if i>0:
                y = self.upsample(y)
            self.attn_outcome_per_group.append(y)

        # # #concatenating multi-group attention
        attn_fused = torch.cat(self.attn_outcome_per_group, axis=1)
        # print(f'attn fused: {attn_fused.shape}')
        attn_fused = attn_fused.reshape(B, self.num_heads, -1, C//self.num_heads)
        attn_fused = attn_fused.permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        # print(f'mh attn: {attn_fused.shape}')
        # exit()
        attn_fused = self.proj(attn_fused)
        attn_fused = self.proj_drop(attn_fused )
        return attn_fused

    def flops(self):
        # FLOPs for linear layers
        flops_linear_q = 2 * self.dim * self.dim
        flops_linear_kv = 2 * self.dim * self.dim * 2
        head_dim = self.dim // self.num_heads
        flops = 0
        print("number of heads ", self.num_heads)
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
    ms_attention = MultiScaleAttention(C, num_heads=4, n_local_region_scales=4, window_size=7, img_size=(56, 56))
    # ms_attention = ms_attention.to(device)
    # # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, H*W, C)
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, H, W)
    print('output: ',y.shape)