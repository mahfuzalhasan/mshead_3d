# Define the WaveletTransform3D module
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


class WaveletTransform3D(nn.Module):
    def __init__(self, wavelet='db1', level=5, mode='zero'):
        super(WaveletTransform3D, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def forward(self, x):
        coeffs = ptwt.wavedec3(x, wavelet=self.wavelet, level=self.level, mode=self.mode)
        Yl = coeffs[0]  # Extracting the approximation coefficients
        return Yl

# Custom function to compute the FLOPs for WaveletTransform3D
def wavelet_transform_flops_wf_1111(input_shape, wavelet, level):
    # Assuming a cubic wavelet filter (e.g., 2x2x2 for `db1`)
    filter_size = 2  # For 'db1', this is a 2-point filter
    C, D, H, W = input_shape

    # Total FLOPs for wavelet transform = #channels * depth * height * width * (filter_size ^ 3)
    total_flops = 0
    for i in range(1, level+1):
        current_d, current_h, current_w = D // (2**i), H // (2**i), W // (2**i)
        flops_per_level = C * current_d * current_h * current_w * (filter_size**3)
        total_flops += flops_per_level

    return total_flops

# Custom function to compute the FLOPs for WaveletTransform3D
def wavelet_transform_flops_wf_3221(input_shape, wavelet, level):
    # Assuming a cubic wavelet filter (e.g., 2x2x2 for `db1`)
    filter_size = 2  # For 'db1', this is a 2-point filter
    C, D, H, W = input_shape

    current_d = D
    current_h = H
    current_w = W
    # Total FLOPs for wavelet transform = #channels * depth * height * width * (filter_size ^ 3)
    total_flops = 0
    for i in range(1, level+1):
        current_d, current_h, current_w = current_d // (2), current_h // (2), current_w // (2)

        flops_per_level = C * current_d * current_h * current_w * (filter_size**3)
        total_flops += flops_per_level

    return total_flops

# Example Input Shape: (Channels, Depth, Height, Width)
input_shape = (48, 48, 48, 48)
flops = wavelet_transform_flops_wf_1111(input_shape, 'db1', 3)
print(f"Approximate FLOPs for WaveletTransform3D: {flops}")
layer_1 = flops * 2

input_shape = (96, 24, 24, 24)
flops = wavelet_transform_flops_wf_1111(input_shape, 'db1', 2)
print(f"Approximate FLOPs for WaveletTransform3D: {flops}")
layer_2 = flops * 2

input_shape = (192, 12, 12, 12)
flops = wavelet_transform_flops_wf_1111(input_shape, 'db1', 1)
print(f"Approximate FLOPs for WaveletTransform3D: {flops}")
layer_3 = flops * 2

total = layer_1 + layer_2 + layer_3

print("total wavelet flops: ", total)



