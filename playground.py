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
# from networks.msHead_3D.network_backbone import MSHEAD_ATTN
# from monai.networks.nets import UNETR, SwinUNETR
import ptwt
import torch

wavelet = 'db1'
level = 3
mode = 'reflect'
B, C, D, H, W = 2, 1, 56, 56, 56
x = torch.randn(B, C, D, H, W)

# Perform wavelet decomposition
coeffs = ptwt.wavedec3(x, wavelet=wavelet, level=level, mode=mode)

# Debugging coefficients structure
print("Type of coeffs:", type(coeffs))
print("Length of coeffs:", len(coeffs))
index = 1
for coeff in coeffs:
    print(f'decompose:{index} -- {type(coeff)}')
    if index == 1:
        print(f'coeff shape: {coeff.shape}')
    for cf in coeff:
        if index == 1:
            print(cf.shape)
        print(type(cf))
    index += 1
# exit()

y1 = coeffs[0]  # Low-frequency coefficients
yh = coeffs[1:]  # High-frequency coefficients
print("Low-frequency shape:", y1.shape)
print("Type of hfs:", type(yh))

corrected_yh = []
for coeff in yh:
    if isinstance(coeff, tuple):
        # Convert tuple to dict
        coeff_dict = {f"{i}": val for i, val in enumerate(coeff)}
        corrected_yh.append(coeff_dict)
    else:
        corrected_yh.append(coeff)

# Perform wavelet reconstruction
yr = ptwt.waverec3((y1, corrected_yh), wavelet=wavelet)
print("Reconstructed shape:", yr.shape)



out_classes = 4

# device = torch.device("cuda")
# print(f'--- device:{device} ---')
# model_1 = UNETR(
#         in_channels=1,
#         out_channels=out_classes,
#         img_size=(96, 96, 96),
#         feature_size=16,
#         hidden_size=768,
#         mlp_dim=3072,
#         num_heads=12,
#         pos_embed="perceptron",
#         norm_name="instance",
#         res_block=True,
#         dropout_rate=0.0,
#     ).to(device)

# data = torch.randn(2, 1, 96, 96, 96)  # Example tensor with B=1, C=3, D=64, H=64, W=64
# data = data.to(device)
# y = model_1(data)
# print(f'y from UNETR: {y.shape}')

# model_2 = SwinUNETR(
#         img_size=(96, 96, 96),
#         in_channels=1,
#         out_channels=out_classes,
#         feature_size=48,
#         use_checkpoint=False,
#     ).to(device)

# y = model_2(data)
# print(f'y from SwinUNETR: {y.shape}')

# model_3 = UXNET(
#         in_chans=1,
#         out_chans=out_classes,
#         depths=[2, 2, 2, 2],
#         feat_size=[48, 96, 192, 384],
#         drop_path_rate=0,
#         layer_scale_init_value=1e-6,
#         spatial_dims=3,
#     ).to(device)

# y = model_3(data)
# print(f'y from 3D UXNET: {y.shape}')
