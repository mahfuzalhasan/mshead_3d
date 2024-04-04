# setting device on GPU if available, else CPU
import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()

# #Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

import torch
# from pytorch_wavelets import DWT3D  # For 2D wavelet transforms
import pywt
import ptwt



# Example input: a batch of 1-channel images, size 64x64
x = torch.randn(1, 1, 64, 64, 64)
transformed = ptwt.wavedec3(x, pywt.Wavelet("haar"), level=2, mode="reflect")
for x in transformed:
    print(type(x))

