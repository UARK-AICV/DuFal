import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Changed from Conv3d to Conv2d
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # Changed from 5D to 4D unpacking
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        # Removed 'd' dimension from rearrange
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        # Removed 'd' dimension from rearrange
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FuseBlock7(nn.Module):
    def __init__(self, channels, num_heads=8):
        super(FuseBlock7, self).__init__()
        # Changed from Conv3d to Conv2d
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels, num_heads=num_heads)
        self.spa_att = Attention(dim=channels, num_heads=num_heads)
        # Changed from Conv3d to Conv2d
        self.fuse = nn.Sequential(
            nn.Conv2d(2*channels, channels, 3, 1, 1),
            nn.Conv2d(channels, 2*channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, spa, fre):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        fre = self.fre_att(fre, spa) + fre
        spa = self.spa_att(spa, fre) + spa
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        # res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res + ori
class FuseBlock8(nn.Module):
    def __init__(self, channels, num_heads=8, fourier_dim=(-2, -1)):
        super(FuseBlock8, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.fourier_dim = fourier_dim
        
        # Convolutional preprocessing for spa and fre
        self.fre = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.spa = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        # Attention modules for real and imaginary components
        self.real_spa_att = Attention(dim=channels, num_heads=num_heads)
        self.imag_spa_att = Attention(dim=channels, num_heads=num_heads)

    def forward(self, spa, fre):
        # Store original spa for residual connection
        ori = spa
        
        # Preprocess spa and fre with convolutions
        spa = self.spa(spa)
        fre = self.fre(fre)
        
        # Step 1: FFT both spa and fre
        spa_fft = torch.fft.fftn(spa, dim=self.fourier_dim)
        fre_fft = torch.fft.fftn(fre, dim=self.fourier_dim)
        
        # Step 2: Extract real and imaginary parts
        real_spa = spa_fft.real
        imag_spa = spa_fft.imag
        real_fre = fre_fft.real
        imag_fre = fre_fft.imag
        
        # Step 3: Update real_spa with attention
        new_real_spa = real_spa + self.real_spa_att(real_spa, real_fre)
        
        # Step 4: Update imag_spa with attention
        new_imag_spa = imag_spa + self.imag_spa_att(imag_spa, imag_fre)
        
        # Step 5: Reconstruct complex tensor and apply IFFT
        spa_updated = torch.complex(new_real_spa, new_imag_spa)
        result = torch.fft.ifftn(spa_updated, dim=self.fourier_dim)
        
        # Take real part of IFFT result for compatibility
        result = result.real
        # Add residual connection
        return result + ori