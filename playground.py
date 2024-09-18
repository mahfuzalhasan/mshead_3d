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
import ptwt
import torch
data = torch.randn(6,16,48,48,48)
for i in range(1, 4):
    coeffs = ptwt.wavedec3(data, wavelet="haar", level=i)
    Y1 = coeffs[0]
    print(Y1.shape)
# print([(key, coeff.shape) for key, coeff in ptwt.wavedec3(data, wavelet="haar", level=2)[-1].items()])

