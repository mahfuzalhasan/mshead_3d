import torch
import torch.nn.functional as F

# def create_haar_filters():
#     # Haar wavelet filters in 3D
#     filters = torch.tensor([
#         [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
#         [[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]],
#         [[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]],
#         [[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]
#     ], dtype=torch.float32) / 8

#     filters = filters.unsqueeze(1)  # Add input channels dimension
#     return filters

# def apply_dwt3d(data, filters):
#     batch_size, channels, depth, height, width = data.shape
#     transformed = []

#     # Apply each filter to each channel separately
#     for i in range(filters.shape[0]):
#         filter_i = filters[i].unsqueeze(1)  # Shape: (1, 1, 2, 2, 2)
#         filter_i = filter_i.repeat(1, channels, 1, 1, 1)  # Repeat filter for each channel
#         # Apply convolution and downsampling
#         conv_result = F.conv3d(data, filter_i, stride=2, padding=1, groups=channels)
#         transformed.append(conv_result)
    
#     # Concatenate all results along the channels dimension
#     transformed = torch.cat(transformed, dim=1)
#     return transformed

# # Example usage
# data = torch.randn(1, 3, 64, 64, 64)  # Example tensor with B=1, C=3, D=64, H=64, W=64

# # Get Haar wavelet filters
# haar_filters = create_haar_filters()

# # Apply 3D DWT
# coeffs = apply_dwt3d(data, haar_filters)

# print("DWT Coefficients shape:", coeffs.shape)




# # Example input: a batch of 1-channel images, size 64x64
# import ptwt
# import torch
# data = torch.randn(6,16,48,48,48)
# for i in range(1, 4):
#     coeffs = ptwt.wavedec3(data, wavelet="haar", level=i)
#     Y1 = coeffs[0]
#     print(Y1.shape)
# # print([(key, coeff.shape) for key, coeff in ptwt.wavedec3(data, wavelet="haar", level=2)[-1].items()])
from networks.UXNet_3D.network_backbone import UXNET
from networks.msHead_3D.network_backbone import MSHEAD_ATTN
from monai.networks.nets import UNETR, SwinUNETR
import ptwt

# wavelet = 'db1'
# level = 1
# mode = 'reflect'
# B, C, D, H, W = 2, 1, 56, 56, 56
# x= torch.randn(B, C, D, H, W)
# coeffs = ptwt.wavedec3(x, wavelet=wavelet, level=level, mode=mode)
# y1 = coeffs[0]
# print(y1.shape)




out_classes = 4

import pywt
import torch

def multi_axis_wavelet_decomposition_torch(tensor, wavelet, levels, axes):
    """
    Perform wavelet decomposition on different axes with different levels in PyTorch.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        wavelet (str): The wavelet to use for decomposition.
        levels (dict): Dictionary specifying the decomposition level for each axis (e.g., {-3: 2, -2: 1, -1: 3}).
        axes (list): List of axes to decompose (e.g., [-3, -2, -1] for D, H, W).

    Returns:
        torch.Tensor: The transformed tensor after applying wavelet decomposition.
    """
    transformed_tensor = tensor

    for axis in axes:
        level = levels.get(axis, 1)  # Default to level 1 if not specified
        # Move the target axis to the last position
        reshaped_tensor = transformed_tensor.transpose(axis, -1)
        # Convert to NumPy for wavelet operations
        reshaped_numpy = reshaped_tensor.cpu().numpy()
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(reshaped_numpy, wavelet, level=level, axis=-1)
        # Keep approximation coefficients (low-frequency part)
        transformed_numpy = coeffs[0]
        # Convert back to PyTorch
        transformed_tensor = torch.from_numpy(transformed_numpy).to(tensor.device)
        # Restore original axis order
        transformed_tensor = transformed_tensor.transpose(axis, -1)

    return transformed_tensor

# Parameters
wavelet = 'db1'
B, C, D, H, W = 2, 1, 14, 56, 56
x = torch.randn(B, C, D, H, W)

# Specify decomposition levels for each axis
levels = {-3: 1, -2: 3, -1: 3}  # Levels for D, H, W

# Decompose tensor
result = multi_axis_wavelet_decomposition_torch(x, wavelet, levels, axes=[-3, -2, -1])
print(f"Transformed tensor shape: {result.shape}")


