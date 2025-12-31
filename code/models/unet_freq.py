from .integral_operators import OperatorBlock_2D
from .hpf import Sobel
import torch.nn as nn
import torch
from einops import rearrange
from .galerkin import simple_attn
import torch.nn.functional as F
from torchinfo import summary
from ptflops import get_model_complexity_info
import math
import numpy as np
class Unet2D_FNO(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_channels: list = [64, 128, 256, 512, 1024], # the first channel is the lifting channel, so the number of stages would be len(num_channels)-1
        target_size: list = [256, 256],
        trunc_mode_stages: list = None, 
        use_sobel_stages: list = None,
        patch_based_stages: list = None,
        patch_size_stages: list = None,
        factorize_mode_stages: list = None,
        use_attn_stages: list = None,
        simple_propagate_stages: list = None,
        window_size_stages: list = None,
        use_swin_stages: list = None,
        include_mid: bool = False, 
        include_up: bool = False,
        skip_type: str = None, # ['add', 'concat']
        type_grid: str = 'linear' # [None, 'linear']
    ):
        super(Unet2D_FNO,self).__init__()
        self.in_channels = in_channels
        self.target_size = target_size
        self.height, self.width = target_size

        # Validate stage configurations
        num_stages = len(num_channels) - 1

        # Validate stage configurations

        assert len(trunc_mode_stages) == num_stages, f"trunc_mode_stages length ({len(trunc_mode_stages)}) must match number of stages ({num_stages})"
        assert len(use_sobel_stages) == num_stages, f"use_sobel_stages length ({len(use_sobel_stages)}) must match number of stages ({num_stages})"
        assert len(patch_based_stages) == num_stages, f"patch_based_stages length ({len(patch_based_stages)}) must match number of stages ({num_stages})"
        assert len(patch_size_stages) == num_stages, f"patch_size_stages length ({len(patch_size_stages)}) must match number of stages ({num_stages})"
        assert len(factorize_mode_stages) == num_stages, f"factorize_mode_stages length ({len(factorize_mode_stages)}) must match number of stages ({num_stages})"
        assert len(use_attn_stages) == num_stages, f"use_attn_stages length ({len(use_attn_stages)}) must match number of stages ({num_stages})"
        assert len(simple_propagate_stages) == num_stages, f"simple_propagate_stages length ({len(simple_propagate_stages)}) must match number of stages ({num_stages})"
        assert len(window_size_stages) == num_stages, f"window_size_stages length ({len(window_size_stages)}) must match number of stages ({num_stages})"
        assert len(use_swin_stages) == num_stages, f"use_swin_stages length ({len(use_swin_stages)}) must match number of stages ({num_stages})"

        self.trunc_mode_stages = trunc_mode_stages
        self.use_sobel_stages = use_sobel_stages
        self.patch_based_stages = patch_based_stages
        self.patch_size_stages = patch_size_stages
        self.factorize_mode_stages = factorize_mode_stages
        self.use_attn_stages = use_attn_stages
        self.simple_propagate_stages = simple_propagate_stages
        self.window_size_stages = window_size_stages
        self.use_swin_stages = use_swin_stages
        self.include_mid = include_mid
        self.include_up = include_up
        self.skip_type = skip_type
        self.type_grid = type_grid
        # Ensure at least one of patch_based, trunc_mode, or simple_propagate is enabled for each stage
        for i in range(num_stages):
            assert (
                patch_based_stages[i] or trunc_mode_stages[i] or simple_propagate_stages[i]
            ), f"Stage {i}: At least one of patch_based_stages[{i}], trunc_mode_stages[{i}], or simple_propagate_stages[{i}] must be True"


        # Initialize layers
        self.use_sobel = any(use_sobel_stages)  # Enable Sobel if any stage uses it
        self.sobel = Sobel() if self.use_sobel else None
        if type_grid == None:
            self.lift00 = nn.Linear(in_channels, num_channels[0]//2)
        else:
            self.lift00 = nn.Linear(in_channels + 2, num_channels[0]//2)
        self.lift01 = nn.Linear(num_channels[0]//2, num_channels[0])
        self.down = nn.ModuleList([])
        for i in range(1, len(num_channels)):
            dim1, dim2 = self.height // 2**(i), self.height // 2**(i)
            if patch_based_stages[i-1]:
                modes1_patch, modes2_patch = patch_size_stages[i-1], patch_size_stages[i-1] // 2+1
            else:
                modes1_patch, modes2_patch = None, None
            if trunc_mode_stages[i-1] == "shared_sliding":
                modes1, modes2 = dim1 // 8 , dim2 // 16 # since weight are shared anyway, use smaller modes
            else:
                modes1, modes2 = dim1 // 2, dim2 // 2
            
            self.down.append(OperatorBlock_2D(
                in_codim=num_channels[i - 1],
                out_codim=num_channels[i],
                dim1=dim1, 
                dim2=dim2,
                modes1=modes1, 
                modes2=modes2,
                modes1_patch=modes1_patch,
                modes2_patch=modes2_patch,
                trunc_mode=trunc_mode_stages[i-1],
                use_attn=use_attn_stages[i-1],
                use_sobel=use_sobel_stages[i-1],
                patch_based=patch_based_stages[i-1],
                patch_size=patch_size_stages[i-1],
                factorize_mode=factorize_mode_stages[i-1],
                simple_propagate=simple_propagate_stages[i-1],
                window_size=window_size_stages[i-1],
                use_swin=use_swin_stages[i-1],
            ))

        # Need to be further inspected, not ready to work
        # Mid layer
        # if self.include_mid:
        #     dim1, dim2 = self.height // 2**(len(num_channels)-1), self.height // 2**(len(num_channels)-1)
        #     modes1, modes2 = dim1 // 2, dim2 // 2
        #     self.mid = OperatorBlock_2D(
        #         in_codim=num_channels[-1],
        #         out_codim=num_channels[-1],
        #         dim1=dim1,
        #         dim2=dim2,
        #         modes1=modes1,
        #         modes2=modes2,
        #         modes1_patch=modes1_patch,
        #         modes2_patch=modes2_patch,
        #         trunc_mode=trunc_mode_stages[-1] if trunc_mode_stages else None,
        #         use_attn=use_attn_stages[-1] if use_attn_stages else False,
        #         use_sobel=use_sobel_stages[-1] if use_sobel_stages else False,
        #         patch_based=patch_based_stages[-1] if patch_based_stages else False,
        #         patch_size=patch_size_stages[-1],
        #         factorize_mode=factorize_mode_stages[-1] if factorize_mode_stages else None,
        #         simple_propagate=simple_propagate_stages[-1] if simple_propagate_stages else False,
        #         window_size=window_size_stages[-1],
        #         use_swin=use_swin_stages[-1],
        #         padding=padding_stages[i-1]
        #     )
        # else:
        #     self.mid = None

        # Upsampling layers
        # self.up = nn.ModuleList([])
        # if self.include_up:
        #     for i in range(len(num_channels)-1, 0, -1):
        #         dim1, dim2 = self.height // 2**(i-1), self.height // 2**(i-1)
        #         if patch_based_stages[i-1]:
        #             modes1_patch, modes2_patch = self.patch_size, self.patch_size // 2 + 1
        #         else:
        #             modes1_patch, modes2_patch = None, None
        #         if trunc_mode_stages[i-1] == "shared_sliding":
        #             modes1, modes2 = dim1 // 16, dim2 // 32
        #         else:
        #             modes1, modes2 = dim1 // 4, dim2 // 4
        #         if skip_type == "add":
        #             in_codim = num_channels[i]
        #         elif skip_type == "concat":
        #             in_codim = num_channels[i] * 2
        #         self.up.append(OperatorBlock_2D(
        #             in_codim=in_codim,
        #             out_codim=num_channels[i-1],
        #             dim1=dim1,
        #             dim2=dim2,
        #             modes1=modes1,
        #             modes2=modes2,
        #             modes1_patch=modes1_patch,
        #             modes2_patch=modes2_patch,
        #             trunc_mode=trunc_mode_stages[i-1],
        #             use_attn=use_attn_stages[i-1],
        #             use_sobel=use_sobel_stages[i-1],
        #             patch_based=patch_based_stages[i-1],
        #             patch_size=patch_size_stages[i-1],
        #             factorize_mode=factorize_mode_stages[i-1],
        #             simple_propagate=simple_propagate_stages[i-1],
        #             window_size=window_size_stages[i-1],
        #             use_swin=use_swin_stages[i-1]
        #         ))
    def forward(self, x): # x: [B*M, C, W, H]
        BM, C, W, H = x.shape
        grid = self.get_grid(x.shape, x.device) if self.type_grid is not None else None
        h = []
        # Sobel filtering
        x_sobel = self.sobel(x) if self.use_sobel else None
        # Concat grid
        if self.type_grid is not None:
            x = torch.cat((x, grid), dim=1)
        # Initial lifting
        x = x.permute(0, 2, 3, 1).contiguous() # x: [B*M, W, H, C]
        x = self.lift00(x) # x: [B*M, W, H, num_channels[0]//2]
        x = F.gelu(x)

        x = self.lift01(x) # x: [B*M, W, H, num_channels[0]//2]
        x = F.gelu(x)
        x = x.permute(0, 3, 1, 2).contiguous() # x: [B*M, num_channels[0], W, H]
        # down
        for i, block in enumerate(self.down):
            x = block(x, x_sobel) if self.use_sobel else block(x)
            h.append(x)
        # Need to be further inspected, not ready to work
        # Mid layer
        # if self.include_mid and self.mid is not None:
        #     x = self.mid(x, x_sobel) if self.use_sobel else self.mid(x)
        #     h.append(x)
        # # Upsampling with skip connections
        # if self.include_up and self.up:
        #     for i, block in enumerate(self.up):
        #         skip = h[-(2 + i)]  # Get corresponding downsampling feature
        #         # Pad if necessary
        #         # diffY = skip.size()[2] - x.size()[2]
        #         # diffX = skip.size()[3] - x.size()[3]
        #         # x = F.pad(x, [diffX // 2, diffX - diffX // 2,
        #         #             diffY // 2, diffY - diffY // 2])
        #         # Concatenate
        #         if self.skip_type == "add":
        #             x += x + skip
        #         elif self.skip_type == "concat":
        #             x = torch.cat([skip, x], dim=1)
        #         x = block(x, x_sobel) if self.use_sobel else block(x)
                
        return x, h
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[-2], shape[-1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

    def get_model_info(self, input_size=None, verbose=0):
        """
        Compute parameters, FLOPs, and memory for the Unet2D_FNO model using torchinfo and ptflops.
        
        Args:
            input_size (tuple, optional): Input shape (e.g., (1, in_channels, H, W)).
                                         Defaults to (1, self.in_channels, target_size[0], target_size[1]).
        
        Returns:
            dict: Dictionary containing:
                - total_params: Total number of parameters.
                - learnable_params: Number of learnable parameters.
                - non_learnable_params: Number of non-learnable parameters.
                - flops_gflops: FLOPs in gigaflops (GFLOPs).
                - memory_mb: Estimated memory in MB for saving state_dict.
                - summary: Detailed layer-wise summary from torchinfo.
        """
        if input_size is None:
            input_size = (1, self.in_channels, self.target_size[0], self.target_size[1])

        # Get parameter counts and summary using torchinfo
        try:
            model_summary = summary(self, input_size=input_size,col_names=["input_size","output_size","num_params"],verbose=verbose)
            total_params = model_summary.total_params
            learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            non_learnable_params = total_params - learnable_params
            summary_str = str(model_summary)
        except Exception as e:
            print(f"Error in torchinfo: {e}")
            total_params = sum(p.numel() for p in self.parameters())
            learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            non_learnable_params = total_params - learnable_params
            summary_str = "Summary unavailable due to error."

        # Get FLOPs using ptflops
        try:
            flops, _ = get_model_complexity_info(
                self,
                input_size[1:],  # Remove batch dimension
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            flops_gflops = flops / 1e9  # Convert to GFLOPs
        except Exception as e:
            print(f"Error in ptflops: {e}")
            flops_gflops = 0.0

        # Estimate memory (4 bytes per parameter, convert to MB)
        memory_mb = total_params * 4 / 1e6  # 4 bytes per parameter, MB = 1e6 bytes

        return {
            "total_params": total_params,
            "learnable_params": learnable_params,
            "non_learnable_params": non_learnable_params,
            "flops_gflops": flops_gflops,
            "memory_mb": memory_mb,
            "summary": summary_str
        }