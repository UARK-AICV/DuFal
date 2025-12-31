import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .galerkin import simple_attn
from einops import rearrange, einsum
class SpectralConv2d_Uno(nn.Module):
    def __init__(
    self,
    in_codim: int,
    out_codim: int,
    dim1: int,
    dim2: int,
    modes1: int = None,
    modes2: int = None,
    trunc_mode: str = None, # None, LL-LH, LH-HH, shared_sliding
    patch_based: bool = False, 
    factorize_mode: str = None # None, dep-sep: Depth-wise Separable Convolution
):    
        super(SpectralConv2d_Uno, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        Ratio of grid size of the input and the output implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2 = Number of fourier modes to consider for the ontegral operator
                        Number of modes must be compatibale with the input grid size 
                        and desired output grid size.
                        i.e., modes1 <= min( dim1/2, input_dim1/2). 
                        Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
                        Other modes also the have same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1
            self.modes2 = modes2
        else:
            self.modes1 = dim1 // 2 - 1
            self.modes2 = dim2 // 2
        self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
        self.trunc_mode = trunc_mode
        self.patch_based = patch_based
        self.factorize_mode = factorize_mode
        # Enforce mutual exclusivity
        if patch_based and trunc_mode is not None:
            raise ValueError("Cannot enable both patch_based=True and trunc_mode is not None")
        assert factorize_mode is None or factorize_mode == "dep-sep", f"factorize_mode should be either real None or str(dep-sep), not {factorize_mode}"
        if patch_based or trunc_mode == "shared_sliding":
            if factorize_mode is None:
                self.weights = nn.Parameter(
                    self.scale * torch.randn(
                        in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                    )
                )
            elif factorize_mode == "dep-sep":
                self.depthwise_weights = nn.Parameter(
                    self.scale * torch.randn(
                        in_codim, self.modes1, self.modes2, dtype=torch.cfloat
                    )
                )
                self.pointwise_weights = nn.Parameter(
                    self.scale * torch.randn(
                        in_codim, out_codim, dtype=torch.cfloat
                    )
                )
        elif trunc_mode in ["LL-LH", "LH-HH"]:
            if factorize_mode is None:
                self.weights1 = nn.Parameter(
                    self.scale * torch.randn(
                        in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                    )
                )
                self.weights2 = nn.Parameter(
                    self.scale * torch.randn(
                        in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                    )
                )
            elif factorize_mode == "dep-sep":
                # Depthwise weights: One set of weights per input channel (in_codim filters, each operating on one input channel)
                self.depthwise_weights1 = nn.Parameter(
                    self.scale * torch.randn(
                        in_codim, self.modes1, self.modes2, dtype=torch.cfloat
                    )
                )
                self.depthwise_weights2 = nn.Parameter(
                    self.scale * torch.randn(
                        in_codim, self.modes1, self.modes2, dtype=torch.cfloat
                    )
                )
                # Pointwise weights: Combine the in_codim channels into out_codim channels (1x1 operation in spectral domain)
                self.pointwise_weights1 = nn.Parameter(
                    self.scale * torch.randn(
                        in_codim, out_codim, dtype=torch.cfloat
                    )
                )
                self.pointwise_weights2 = nn.Parameter(
                    self.scale * torch.randn(
                        in_codim, out_codim, dtype=torch.cfloat
                    )
                )

    # Complex multiplication
    def compl_mul2d(self, inputs, weights, depthwise_weights=None, pointwise_weights=None):
        if self.trunc_mode == "shared_sliding" :
            # Non-overlapping sliding mode
            batchsize, channel, inputs_dim1, inputs_dim2 = inputs.shape
            numpx = inputs_dim1 // self.modes1 # number of patches along x dim
            numpy = inputs_dim2 // self.modes2 # number of patches along y dim
            # Assert input dimension are divisible by modes
            assert inputs_dim1 % self.modes1 == 0, f"input_dim1 ({inputs_dim1}) not divisible for ({self.modes1})"
            assert inputs_dim2 % self.modes2 == 0, f"input_dim1 ({inputs_dim2}) not divisible for ({self.modes2})"
            # Reshape inputs into patches
            inputs_patched = rearrange(
                inputs,
                'b c (numpx sizepx) (numpy sizepy) -> (b numpx numpy) c sizepx sizepy',
                numpx=numpx,
                numpy=numpy,
                sizepx=self.modes1,
                sizepy=self.modes2
            )
            if self.factorize_mode is None:
                out = einsum(
                    inputs_patched,
                    weights,
                    'b_numpx_numpy in_c sizepx sizepy, in_c out_c sizepx sizepy -> b_numpx_numpy out_c sizepx sizepy'
                )
            elif self.factorize_mode == "dep-sep":
                out_depthwise = einsum(
                    inputs_patched,
                    depthwise_weights,
                    'b_numpx_numpy in_c sizepx sizepy, in_c sizepx sizepy -> b_numpx_numpy in_c sizepx sizepy'
                )
                out = einsum(
                    out_depthwise,
                    pointwise_weights,
                    'b_numpx_numpy in_c sizepx sizepy, in_c out_c -> b_numpx_numpy out_c sizepx sizepy'
                )
            out = rearrange(
                out,
                '(b numpx numpy) out_c sizepx sizepy -> b out_c (numpx sizepx) (numpy sizepy)',
                numpx=numpx,
                numpy=numpy,
            )
            return out
        else:
            if self.factorize_mode is None:
                # For non-sliding modes, keep the original einsum
                out = einsum(
                    inputs,
                    weights,
                    'b i x y, i o x y-> b o x y'
                )
                return out
            elif self.factorize_mode == "dep-sep":
                out_depthwise = einsum(
                    inputs,
                    depthwise_weights,
                    'b i x y, i x y -> b i x y'
                )
                out = einsum(
                    out_depthwise,
                    pointwise_weights,
                    'b i x y, i o -> b o x y'
                )
                return out

    def forward(self, x, dim1=None, dim2=None):
        if dim1 is None or dim2 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        batchsize, in_channel, input_dim1, input_dim2 = x.shape

        if not self.patch_based:
            # Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = torch.fft.rfft2(x, norm="forward")
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                input_dim1,
                input_dim2 // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            if self.trunc_mode == "LL-LH":
                if self.factorize_mode is None:
                    out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
                        x_ft[:, :, :self.modes1, :self.modes2], weights=self.weights1
                    )
                    out_ft[:, :, :self.modes1, -self.modes2: ] = self.compl_mul2d(
                        x_ft[:, :, :self.modes1, -self.modes2: ], weights=self.weights2
                    )
                elif self.factorize_mode == "dep-sep":
                    out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
                        x_ft[:, :, :self.modes1, :self.modes2], weights=None, 
                        depthwise_weights=self.depthwise_weights1, pointwise_weights=self.pointwise_weights1
                    )
                    out_ft[:, :, :self.modes1, -self.modes2: ] = self.compl_mul2d(
                        x_ft[:, :, :self.modes1, -self.modes2: ], weights=None, 
                        depthwise_weights=self.depthwise_weights2, pointwise_weights=self.pointwise_weights2
                    )
            elif self.trunc_mode == "LH-HH":
                if self.factorize_mode is None:
                    out_ft[:, :, :self.modes1, -self.modes2:] = self.compl_mul2d(
                        x_ft[:, :, :self.modes1, -self.modes2:], weights=self.weights1
                    )
                    out_ft[:, :, -self.modes1:, -self.modes2: ] = self.compl_mul2d(
                        x_ft[:, :, -self.modes1:, -self.modes2: ], weights=self.weights2
                    )
                elif self.factorize_mode == "dep-sep":
                    out_ft[:, :, :self.modes1, -self.modes2:] = self.compl_mul2d(
                        x_ft[:, :, :self.modes1, -self.modes2:], weights=None, 
                        depthwise_weights=self.depthwise_weights1, pointwise_weights=self.pointwise_weights1
                    )
                    out_ft[:, :, -self.modes1:, -self.modes2: ] = self.compl_mul2d(
                        x_ft[:, :, -self.modes1:, -self.modes2: ], weights=None, 
                        depthwise_weights=self.depthwise_weights2, pointwise_weights=self.pointwise_weights2
                    )
            elif self.trunc_mode == "shared_sliding":
                if self.factorize_mode is None:
                    out_ft[:, :, :input_dim1, :(input_dim2 // 2)] = self.compl_mul2d(x_ft[:, :, :input_dim1, :(input_dim2 // 2)], weights=self.weights) # no +1 cause we deliberately omit the last column of last dim
                elif self.factorize_mode == "dep-sep":
                    out_ft[:, :, :input_dim1, :(input_dim2 // 2)] = self.compl_mul2d(x_ft[:, :, :input_dim1, :(input_dim2 // 2)], weights=None, 
                    depthwise_weights=self.depthwise_weights, pointwise_weights=self.pointwise_weights) # no +1 cause we deliberately omit the last column of last dim
            # Return to physical space
            x = torch.fft.irfft2(out_ft, s=(input_dim1, input_dim2), norm="forward")
            if self.dim1 != input_dim1 or self.dim2 != input_dim2:
                x = F.interpolate(
                    x, size=(self.dim1, self.dim2), mode='bicubic', align_corners=True, antialias=True
                )
            return x
        elif self.patch_based:
            # now each input is a batch of patches : B*M*num_patches, C, patch_size, patch_size
            x_ft = torch.fft.rfft2(x, norm="forward")
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                input_dim1,
                input_dim2 // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            if self.factorize_mode is None:
                out_ft = self.compl_mul2d(x_ft, weights=self.weights)
            elif self.factorize_mode == "dep-sep":
                out_ft = self.compl_mul2d(x_ft, weights=None, depthwise_weights=self.depthwise_weights, pointwise_weights=self.pointwise_weights)
            x = torch.fft.irfft2(out_ft, s=(input_dim1, input_dim2), norm="forward")
            return x


            




class pointwise_op_2D(nn.Module):
    """
    dim1 = Default output grid size along x (or 1st dimension)
    dim2 = Default output grid size along y ( or 2nd dimension)
    in_codim = Input co-domian dimension
    out_codim = output co-domain dimension
    """

    def __init__(self, in_codim, out_codim, dim1, dim2, patch_based):
        super(pointwise_op_2D, self).__init__()
        self.patch_based = patch_based
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self, x, dim1=None, dim2=None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        if dim1 is None or dim2 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)

        # ft = torch.fft.rfft2(x_out)
        # ft_u = torch.zeros_like(ft)
        # ft_u[:dim1//2-1,:dim2//2-1] = ft[:dim1//2-1,:dim2//2-1]
        # ft_u[-(dim1//2-1):,:dim2//2-1] = ft[-(dim1//2-1):,:dim2//2-1]
        # x_out = torch.fft.irfft2(ft_u)
        if not self.patch_based:
            x_out = torch.nn.functional.interpolate(
                x_out, size=(dim1, dim2), mode="bicubic", align_corners=True, antialias=True
            )
        return x_out


class OperatorBlock_2D(nn.Module):
    """
    Normalize = if true performs InstanceNorm2d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv2d_Uno class.
    """

    def __init__(
        self,
        in_codim,           # Number of input channels
        out_codim,          # Number of output channels
        dim1,               # Height of the input feature map
        dim2,               # Width of the input feature map
        modes1,             # Number of Fourier modes for height in trunc_mode
        modes2,             # Number of Fourier modes for width in trunc_mode
        modes1_patch,       # Number of Fourier modes for height in patch-based mode
        modes2_patch,       # Number of Fourier modes for width in patch-based mode
        trunc_mode=None,    # Type of truncation mode for spectral convolution, if any
        Normalize=True,     # Whether to apply InstanceNorm2d on the output
        Non_Lin=True,       # Whether to apply point-wise nonlinearity
        num_heads=8,        # Number of attention heads for attention mechanisms
        use_attn=False,     # Whether to use attention mechanism
        use_sobel=False,    # Whether to apply Sobel convolution
        patch_based=False,  # Whether to use patch-based spectral convolution
        patch_size=16,      # Size of patches for patch-based processing
        factorize_mode=None,# Mode for factorization in spectral convolution
        simple_propagate=False,  # Whether to use simple propagation (MLP + interpolation)
        window_size=2,      # Number of patches per window dimension for Swin Transformer
        use_swin=True,      # Whether to use Swin Transformer-inspired layers
    ):
        super(OperatorBlock_2D, self).__init__()

        # Initialize parameters
        self.in_codim = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes1_patch = modes1_patch
        self.modes2_patch = modes2_patch
        self.trunc_mode = trunc_mode
        self.normalize = Normalize
        self.non_lin = Non_Lin
        self.num_heads = num_heads
        self.use_attn = use_attn
        self.use_sobel = use_sobel
        self.patch_based = patch_based
        self.patch_size = patch_size
        self.factorize_mode = factorize_mode
        self.simple_propagate = simple_propagate
        self.window_size = window_size
        self.use_swin = use_swin

        assert trunc_mode is not None or patch_based or simple_propagate, "Either trunc_mode or patch_based or simple_propagate must be enable"
        # Initialize SpectralConv2d_Uno for trunc_mode if enabled
        if trunc_mode is not None:
            self.conv_trunc = SpectralConv2d_Uno(
                in_codim, out_codim, dim1, dim2, modes1, modes2, trunc_mode=trunc_mode, patch_based=False, factorize_mode=factorize_mode
            )
        else:
            self.conv_trunc = None

        # Initialize SpectralConv2d_Uno for patch_based if enabled
        if patch_based:
            self.conv_patch = SpectralConv2d_Uno(
                in_codim, out_codim, dim1, dim2, modes1_patch, modes2_patch, trunc_mode=None, patch_based=True, factorize_mode=factorize_mode
            )
        else:
            self.conv_patch = None

        # Pointwise operator
        self.w = pointwise_op_2D(in_codim, out_codim, dim1, dim2, patch_based=None)
        # Simple propagation branch (MLP + interpolation)
        if simple_propagate:
            self.simple_mlp = nn.Conv2d(int(in_codim), int(out_codim), 1)  # MLP-like channel transformation
        else:
            self.simple_mlp = None
        # Swin Transformer-inspired layers
        if use_swin and patch_based:
            self.wo_shift_SA = simple_attn(out_codim, heads=num_heads)
            self.w_shift_SA = simple_attn(out_codim, heads=num_heads)
        else:
            self.wo_shift_SA = None
            self.w_shift_SA = None
        # Attention, normalization, and non-linearity
        if use_attn:
            self.all_branch_attn = simple_attn(out_codim, heads=num_heads)
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim), affine=True)
        # Sobel convolution
        if use_sobel:
            self.sobel_conv = nn.Conv2d(1, out_codim, kernel_size=1)
    
    def _patch_split(self, x: torch.Tensor) -> torch.Tensor:
        """Split input into patches."""
        BM, C, W, H = x.shape
        n_patch_w = W // self.patch_size
        n_patch_h = H // self.patch_size
        output = rearrange(
            x,
            "BM C (n_patch_w s_patch_w) (n_patch_h s_patch_h) -> "
            "(BM n_patch_w n_patch_h) C s_patch_w s_patch_h",
            n_patch_w=n_patch_w,
            n_patch_h=n_patch_h,
            s_patch_w=self.patch_size,
            s_patch_h=self.patch_size
        )
        return output
    def _window_partition_patches(self, x_patches: torch.Tensor, BM: int, n_patch_w: int, n_patch_h: int, shift: bool = False) -> torch.Tensor:
        """Partition patches into windows, optionally with shift."""
        # Input: (B*num_patch_w*num_patch_h) x C x size_patch_w x size_patch_h
        # Reshape to separate batch and patch dimensions
        x_patches = rearrange(
            x_patches,
            "(BM n_patch_w n_patch_h) C s_patch_w s_patch_h -> BM C s_patch_w s_patch_h n_patch_w n_patch_h",
            BM=BM,
            n_patch_w=n_patch_w,
            n_patch_h=n_patch_h
        )

        if shift:
            # Shift the patch grid by half the window size (in patch units)
            shift_w = shift_h = self.window_size // 2
            x_patches = torch.roll(x_patches, shifts=(-shift_w, -shift_h), dims=(4, 5))

        # Calculate number of windows
        n_window_w = n_patch_w // self.window_size
        n_window_h = n_patch_h // self.window_size

        # Partition into windows
        x_windows = rearrange(
            x_patches,
            "BM C s_patch_w s_patch_h (n_window_w s_window_w) (n_window_h s_window_h) -> "
            "(BM n_window_w n_window_h) C s_window_w s_window_h s_patch_w s_patch_h",
            n_window_w=n_window_w,
            n_window_h=n_window_h,
            s_window_w=self.window_size,
            s_window_h=self.window_size
        )

        # Naively merge patches within each window
        x_windows = rearrange(
            x_windows,
            "BM_and_n_window_w_and_n_window_h C s_window_w s_window_h s_patch_w s_patch_h -> "
            "BM_and_n_window_w_and_n_window_h C (s_window_w s_patch_w) (s_window_h s_patch_h)",
            s_window_w=self.window_size,
            s_window_h=self.window_size,
            s_patch_w=self.patch_size,
            s_patch_h=self.patch_size
        ) # N_window x C x W x H, with W,H is pixel-unit size of a window (not patch-unit)
        
        return x_windows, n_window_w, n_window_h

    def _window_reverse_patches(self, x_windows: torch.Tensor, BM: int, n_patch_w: int, n_patch_h: int, n_window_w: int, n_window_h: int, shift: bool = False) -> torch.Tensor:
        """Reverse window partitioning, optionally undoing shift, and reconstruct patched form."""
        # Input: (B*num_window_w*num_window_h) x C x (size_window_w*size_patch_w) x (size_window_h*size_patch_h)
        # Reshape to separate window and patch dimensions
        x_windows = rearrange(
            x_windows,
            "BM_and_n_window_w_and_n_window_h C (s_window_w s_patch_w) (s_window_h s_patch_h) -> "
            "BM_and_n_window_w_and_n_window_h C s_window_w s_window_h s_patch_w s_patch_h",
            s_window_w=self.window_size,
            s_window_h=self.window_size,
            s_patch_w=self.patch_size,
            s_patch_h=self.patch_size
        )

        # Reverse window partitioning
        x_patches = rearrange(
            x_windows,
            "(BM n_window_w n_window_h) C s_window_w s_window_h s_patch_w s_patch_h -> "
            "BM C s_patch_w s_patch_h (n_window_w s_window_w) (n_window_h s_window_h)",
            BM=BM,
            n_window_w=n_window_w,
            n_window_h=n_window_h
        )

        if shift:
            # Undo shift
            shift_w = shift_h = self.window_size // 2
            x_patches = torch.roll(x_patches, shifts=(shift_w, shift_h), dims=(4, 5))

        # Reshape back to patched form
        x_patches = rearrange(
            x_patches,
            "BM C s_patch_w s_patch_h n_patch_w n_patch_h -> "
            "(BM n_patch_w n_patch_h) C s_patch_w s_patch_h",
            n_patch_w=n_patch_w,
            n_patch_h=n_patch_h
        )

        return x_patches

    def forward(self, x, x_sobel=None, dim1=None, dim2=None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        if dim1 is None or dim2 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        BM, in_channel, input_dim1, input_dim2 = x.shape
        # Initialize output
        x_out = torch.zeros(
            BM, self.out_channels, dim1, dim2,
            device=x.device, dtype=x.dtype
        )
        # Truncation mode branch
        if self.conv_trunc is not None:
            x_out += self.conv_trunc(x, dim1, dim2)

        # Patch-based branch
        if self.conv_patch is not None:
            x_patch = self._patch_split(x) # (BM*num_patch^2, C, patch_size, patch_size)
            x_patch = self.conv_patch(x_patch, dim1, dim2)
            n_patch_w = n_patch_h = int((x_patch.shape[0] // BM) ** 0.5)
            
            # Apply Swin-style attention:
            if self.use_swin and self.wo_shift_SA is not None and self.w_shift_SA is not None:
  
                # Regular window attention (no shift)
                x_patch, n_window_w, n_window_h = self._window_partition_patches(
                    x_patch, BM, n_patch_w, n_patch_h, shift=False
                ) 
                # (BM n_window_w n_window_h) C (s_window_w s_patch_w) (s_window_h s_patch_h)
                # total_number_window x C x window_size_w[pixel-level] x window_size_h[pixel-level]
                x_patch = self.wo_shift_SA(x_patch)  # Self-attention within windows
                x_patch = self._window_reverse_patches(
                    x_patch, BM, n_patch_w, n_patch_h, n_window_w, n_window_h, shift=False
                )
                # Shifted window attention
                x_patch, n_window_w, n_window_h = self._window_partition_patches(
                    x_patch, BM, n_patch_w, n_patch_h, shift=True
                )
                x_patch = self.w_shift_SA(x_patch)  # Self-attention within windows
                x_patch = self._window_reverse_patches(
                    x_patch, BM, n_patch_w, n_patch_h, n_window_w, n_window_h, shift=True
                )



            # Reconstruct from the patched form
            x_patch = rearrange(x_patch, 
            "(BM n_patch_w n_patch_h) C s_patch_w s_patch_h -> BM C (n_patch_w s_patch_w) (n_patch_h s_patch_h)", 
                BM=BM, 
                n_patch_w=n_patch_w, 
                n_patch_h=n_patch_h, 
                s_patch_h=self.patch_size, 
                s_patch_w=self.patch_size
            )
            # Interpolate to desired size
            x_patch = F.interpolate(
                x_patch, size=(dim1, dim2), mode='bicubic', align_corners=True, antialias=True
            )
            x_out += x_patch

        # Pointwise operator
        x_out += self.w(x, dim1, dim2)

        # Simple propagation branch
        if self.simple_mlp is not None:
            x_simple = self.simple_mlp(x)  # Channel transformation
            x_simple = F.interpolate(
                x_simple, size=(dim1, dim2), mode='bicubic', align_corners=True, antialias=True
            )
            x_out += x_simple

        # Sobel edge detection
        if self.use_sobel and x_sobel is not None:
            x_sobel_out = self.sobel_conv(x_sobel)
            if x_sobel_out.shape[2] != dim1 or x_sobel_out.shape[3] != dim2:
                x_sobel_out = F.interpolate(
                    x_sobel_out, size=(dim1, dim2), mode='bicubic', align_corners=True, antialias=True
                )
            x_out += x_sobel_out

        # Normalization
        if self.normalize:
            x_out = self.normalize_layer(x_out)

        # Non-linearity
        if self.non_lin:
            x_out = F.gelu(x_out)

        # Attention
        if self.use_attn:
            residual = x_out
            x_out = self.all_branch_attn(x_out)
            x_out += residual

        return x_out