import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange  # NEW: Added einops import for rearrange
def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer
class Sobel(nn.Module):  # NEW: Added Sobel operator class
    def __init__(self):
        super().__init__()
        # Convolutional layer for Sobel filters: 1 input channel, 2 output channels (Gx and Gy), 3x3 kernel
        # padding=1 ensures output spatial dimensions match input
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        # Sobel kernels for horizontal (Gx) and vertical (Gy) gradients
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])  # Shape: [3, 3]
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])  # Shape: [3, 3]
        # Combine Gx and Gy into a single tensor: [2, 3, 3]
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        # Add channel dimension for Conv2d: [2, 1, 3, 3] (2 filters, 1 input channel, 3x3 kernel)
        G = G.unsqueeze(1)
        # Set convolution weights as non-trainable
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        # Input shape: [B, 1, H, W] (batch size B, 1 channel, height H, width W)
        x = self.filter(img)  # Apply Sobel filters (Gx, Gy)
        # Shape: [B, 2, H, W] (2 channels for Gx and Gy, spatial dimensions preserved due to padding=1)
        
        x = torch.mul(x, x)  # Element-wise square: (Gx)^2, (Gy)^2
        # Shape: [B, 2, H, W] (same shape, values are squared)
        
        x = torch.sum(x, dim=1, keepdim=True)  # Sum across channels: (Gx)^2 + (Gy)^2
        # Shape: [B, 1, H, W] (1 channel, sum of squared gradients)
        
        x = torch.sqrt(x)  # Take square root to get gradient magnitude: sqrt((Gx)^2 + (Gy)^2)
        # Shape: [B, 1, H, W] (1 channel, gradient magnitude)
        
        return x
class Downsample_PASA_group_softmax(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(Downsample_PASA_group_softmax, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        # CHANGED: Replaced sigma.reshape(n,1,c,h*w) with einops.rearrange
        sigma = rearrange(sigma, 'n c h w -> n 1 c (h w)')

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        # CHANGED: Replaced x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)
        # with einops.rearrange
        x = rearrange(x, 'n c1 p q -> c1 n p q')  # permute(1,0,2,3)
        x = rearrange(x, '(g c1g) n p q -> n g c1g p q', g=self.group)  # reshape(self.group, c1//self.group, n, p, q)
        # Note: c1 is split into group and c1//group as (g c1g)

        n,c2,p,q = sigma.shape
        # CHANGED: Replaced sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size, n, c2, q)).permute(2,0,3,1,4)
        # with einops.rearrange
        sigma = rearrange(sigma, 'n c2 p q -> p n c2 q')  # permute(2,0,1,3)
        sigma = rearrange(sigma, '(pk k2) n c2 q -> n pk c2 k2 q', k2=self.kernel_size*self.kernel_size)  # reshape
        # Note: p is split as (pk k2) where k2 is kernel_size*kernel_size
        sigma = rearrange(sigma, 'n pk c2 k2 q -> n pk c2 k2 q')  # Already in desired order

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]

class Downsample_PASA_group_softmax_hpf(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(Downsample_PASA_group_softmax_hpf, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        # Create an identity kernel (center is 1, others are 0)
        identity_kernel = torch.zeros((kernel_size * kernel_size,))
        center_idx = (kernel_size * kernel_size) // 2  # Center index for a flattened kxk kernel
        identity_kernel[center_idx] = 1.0
        self.register_buffer('identity_kernel', identity_kernel)

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma) # Shape: (n, group * k * k, h, w)

        n,c,h,w = sigma.shape
        # CHANGED: Replaced sigma.reshape(n, self.group, self.kernel_size * self.kernel_size, h, w)
        # with einops.rearrange
        sigma = rearrange(sigma, 'n (g k2) h w -> n g k2 h w', k2=self.kernel_size * self.kernel_size)
        sigma = self.identity_kernel.view(1, 1, self.kernel_size * self.kernel_size, 1, 1) - sigma # Shape: (n, group, k*k, h, w)

        # CHANGED: Replaced sigma.reshape(n,1,c,h*w) with einops.rearrange
        sigma = rearrange(sigma, 'n g k2 h w -> n 1 (g k2) (h w)')

        n,c,h,w = x.shape
        x = rearrange(
            F.unfold(self.pad(x), kernel_size=self.kernel_size),
            'n (c k2) (h w) -> n c k2 (h w)',
            c=c,
            k2=self.kernel_size * self.kernel_size,
            h=h,
            w=w
        )

        n,c1,p,q = x.shape
        # CHANGED: Replaced x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)
        # with einops.rearrange
        x = rearrange(x, 'n c1 p q -> c1 n p q')  # permute(1,0,2,3)
        x = rearrange(x, '(g c1g) n p q -> n g c1g p q', g=self.group)  # reshape(self.group, c1//self.group, n, p, q)
        # Note: c1 is split into group and c1//group as (g c1g)

        n,c2,p,q = sigma.shape
        # CHANGED: Replaced sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size, n, c2, q)).permute(2,0,3,1,4)
        # with einops.rearrange
        sigma = rearrange(sigma, 'n c2 p q -> p n c2 q')  # permute(2,0,1,3)
        sigma = rearrange(sigma, '(pk k2) n c2 q -> n pk c2 k2 q', k2=self.kernel_size * self.kernel_size)  # reshape
        # Note: p is split as (pk k2) where k2 is kernel_size*kernel_size

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]

# Test script
def test_models():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Hyperparameters
    batch_size = 2
    in_channels = 64
    height = 32
    width = 32
    kernel_size = 3
    stride = 1
    group = 2

    # Create input tensor
    x = torch.randn(batch_size, in_channels, height, width)

    # Initialize models
    model1 = Downsample_PASA_group_softmax(
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        pad_type='reflect',
        group=group
    )
    model2 = Downsample_PASA_group_softmax_hpf(
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        pad_type='reflect',
        group=group
    )

    # Set models to evaluation mode
    model1.eval()
    model2.eval()

    # Run forward pass
    with torch.no_grad():
        output1 = model1(x)
        output2 = model2(x)

    # Print shapes and basic statistics
    print("Input shape:", x.shape)
    print("Downsample_PASA_group_softmax output shape:", output1.shape)
    print("Downsample_PASA_group_softmax_hpf output shape:", output2.shape)
    print("\nOutput statistics:")
    print("Model 1 mean:", output1.mean().item(), "std:", output1.std().item())
    print("Model 2 mean:", output2.mean().item(), "std:", output2.std().item())

    # Check output shapes
    expected_h = height // stride
    expected_w = width // stride
    expected_shape = (batch_size, in_channels, expected_h, expected_w)
    assert output1.shape == expected_shape, f"Model 1 output shape mismatch: {output1.shape} vs {expected_shape}"
    assert output2.shape == expected_shape, f"Model 2 output shape mismatch: {output2.shape} vs {expected_shape}"

    print("\nAll shape tests passed!")

# Run the test
if __name__ == "__main__":
    test_models()