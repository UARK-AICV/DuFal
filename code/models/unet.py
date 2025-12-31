import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_freq import Unet2D_FNO
from .fuse_block import FuseBlock7, FuseBlock8
class UNet(nn.Module):
    def __init__(self, 
                 in_ch: int = 1,
                 out_ch: int = 128,
                 trunc_mode_stages: list = None,
                 use_sobel_stages: list = None,
                 patch_based_stages: list = None,
                 patch_size_stages: list = None,
                 factorize_mode_stages: list = None,
                 use_attn_stages: list = None,
                 simple_propagate_stages: list = None,
                 window_size_stages: list = None,
                 use_swin_stages: list = None,
                 use_fno_only: bool = False,
                 skip_type: str = None,
                 type_grid: str = None, 
                 fuse_block: int = None):
        """
        Initializes the UNet model with customizable architecture options.

        Args:
            in_ch (int): Number of input channels (default: 1)
            out_ch (int): Number of output channels (default: 128)
            trunc_mode_stages (list): List specifying truncation modes for each stage
                                    (e.g., ["LL-LH", "LH-HH", "shared_sliding"])
                                    None means no truncation applied
            use_sobel_stages (list): List of booleans indicating whether to use Sobel
                                   filters at each stage
            patch_based_stages (list): List of booleans indicating whether to use
                                     patch-based processing at each stage
            patch_size (int): Size of patches for patch-based processing (default: 16)
            factorize_mode_stages (list): List specifying factorization modes for each stage
            use_attn_stages (list): List of booleans indicating whether to use attention
                                  mechanisms at each stage
            simple_propagate_stages (list): List of booleans indicating whether to use
                                         simple propagation at each stage
            window_size_stages (list): List of window sizes for attention mechanisms
                                     at each stage
            use_swin_stages (list): List of booleans indicating whether to use Swin
                                  Transformer at each stage
            use_fno_only (bool): If True, uses only FNO (Fourier Neural Operator)
                               architecture (default: False)
            type_grid (str): Type of grid to use (e.g., None, 'linear')
                           None means don't use grid
            fuse_block (int): Fuse block used
        """
        super().__init__()

        self._out_ch = out_ch
        self._ds_ch = 1024
        self.use_fno_only = use_fno_only
        self.fuse_block = fuse_block

        if not use_fno_only:
            self.inc = DoubleConv(in_ch, 64)
            self.fuse = nn.ModuleList()
            if self.fuse_block == 7:
                get_fuse_block = FuseBlock7
            elif self.fuse_block == 8:
                get_fuse_block = FuseBlock8
            for m in [
                get_fuse_block(128, num_heads=8),
                get_fuse_block(256, num_heads=8),
                get_fuse_block(512, num_heads=8),
                get_fuse_block(1024, num_heads=8),
            ]:
                self.fuse.append(m)
            self.down = nn.ModuleList()
            for m in [           # in: 256x
                DownConv(64, 128),   # 128x
                DownConv(128, 256),  # 64x
                DownConv(256, 512),  # 32x
                DownConv(512, 1024)  # 16x
            ]:
                self.down.append(m)

            self.up = nn.ModuleList()
            for m in [
                UpConv(1024, 512), # 32x
                UpConv(512, 256),  # 64x
                UpConv(256, 128),  # 128x
                UpConv(128, 64)    # 256x
            ]:
                self.up.append(m)

        self.outc = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

        self.unet_freq = Unet2D_FNO(in_channels=1, 
                                    num_channels=[64, 128, 256, 512, 1024], 
                                    target_size=[256, 256], 
                                    trunc_mode_stages=trunc_mode_stages, 
                                    use_sobel_stages=use_sobel_stages, 
                                    patch_based_stages=patch_based_stages,
                                    patch_size_stages=patch_size_stages, 
                                    factorize_mode_stages=factorize_mode_stages,
                                    use_attn_stages=use_attn_stages,
                                    simple_propagate_stages=simple_propagate_stages,
                                    window_size_stages=window_size_stages,
                                    use_swin_stages=use_swin_stages,
                                    include_mid=True if use_fno_only else False,
                                    include_up=use_fno_only, # Enable upsampling only when FNO is standalone)
                                    skip_type=skip_type,
                                    type_grid=type_grid)  
    @property
    def out_ch(self):
        return self._out_ch

    @property
    def ds_ch(self):
        return self._ds_ch

    def forward(self, x):
        # x: [B, M, C, W, H], M: the number of views
        # outputs: [B, M, C', W', H']
        b, m = x.shape[:2]
        x = x.reshape(b * m, *x.shape[2:]) # [B*M, C, W, H]

        if self.use_fno_only:
            # FNO-only forward pass
            y, h = self.unet_freq(x)  # y is the final output, h contains hidden states
            y = self.outc(y)  # Apply outc to FNO output
            ds_x = h[-1]  # Use mid-layer (or last downsampling output if no mid-layer)
            return {
                'feats': y.reshape(b, m, *y.shape[1:]),
                'feats_ds': ds_x.reshape(b, m, *ds_x.shape[1:])
            }

        # freq
        fre_x1_s, fre_h_s = self.unet_freq(x)

        x1 = self.inc(x)

        xs = [x1]
        for (conv, fuse, fre_h) in zip(self.down, self.fuse, fre_h_s):
            xs.append(fuse(conv(xs[-1]),fre_h))

        x = xs[-1]
        ds_x = x

        for i, conv in enumerate(self.up):
            x = conv(x, xs[-(2 + i)])

        y = self.outc(x)
        return {
            'feats': y.reshape(b, m, *y.shape[1:]), 
            'feats_ds': ds_x.reshape(b, m, *ds_x.shape[1:])
        }
    # def forward(self, x):
    #     # x: [B, M, C, W, H], M: the number of views
    #     # outputs: [B, M, C', W', H']
    #     b, m = x.shape[:2]
    #     print(f"Input shape: {x.shape}")
        
    #     x = x.reshape(b * m, *x.shape[2:])
    #     print(f"After reshape: {x.shape}")

    #     # Initial convolution
    #     x1 = self.inc(x)
    #     print(f"After initial conv (inc): {x1.shape}")

    #     # Downward path
    #     xs = [x1]
    #     for i, conv in enumerate(self.down):
    #         xs.append(conv(xs[-1]))
    #         print(f"After down conv {i+1}: {xs[-1].shape}")

    #     # Bottom of U-Net
    #     x = xs[-1]
    #     ds_x = x
    #     print(f"Bottom of U-Net: {x.shape}")

    #     # Upward path
    #     for i, conv in enumerate(self.up):
    #         x = conv(x, xs[-(2 + i)])
    #         print(f"After up conv {i+1}: {x.shape}")

    #     # Output convolution
    #     y = self.outc(x)
    #     print(f"After output conv: {y.shape}")

    #     # Final reshaping
    #     feats = y.reshape(b, m, *y.shape[1:])
    #     feats_ds = ds_x.reshape(b, m, *ds_x.shape[1:])
    #     print(f"Final feats shape: {feats.shape}")
    #     print(f"Final feats_ds shape: {feats_ds.shape}")
    #     fuiodsf
    #     return {
    #         'feats': feats,
    #         'feats_ds': feats_ds
    #     }


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        '''
           if you have padding issues, see
           https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
           https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        '''
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


