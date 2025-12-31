# DuFal
## DuFal: Dual-Frequency-Aware Learning for High-Fidelity Extremely Sparse-View CBCT Reconstruction

Cuong Tran, Trong-Thang Pham, Son Nguyen, Duy Minh Ho Nguyen, Ngan Le

Codebase for DuFal, an end-to-end sparse-view CBCT reconstruction framework that integrates spatial and frequency-domain processing to preserve fine anatomical details under extremely sparse-view settings.

## Highlights
- Dual-path encoding with spatial and frequency branches for high-frequency detail preservation.
- HiLocFFNO blocks combine global and local frequency modeling in a modular Frequency Encoder.
- Spectral-Channel Factorization (SCF) reduces FNO parameters while retaining quality.
- Cross-Attention Frequency Fusion (CAFF) merges spatial and spectral features in the frequency domain.
- Evaluated on LUNA16 and ToothFairy in extremely sparse-view settings.

## Installation
```bash
pip install torch==1.13 pytorch3d SimpleITK easydict
```

## Configuration
Edit `configs/config.yaml` to set your data path and model parameters:

```yaml
dataset:
  root_dir: /path/to/your/data
  gs_res: 12

model:
  image_encoder:
    out_ch: 128
  point_decoder:
    mlp_chs: [256, 64, 16, 1]
  gs:
    t: 512        # temperature scaling factor
    res: 12       # resolution of Gaussian splatting points
    p_dist_scaling: 1.0
    o_scaling: 1.0
    s_scaling: 0.3
```

## Usage
### Evaluation
```bash
python code/evaluate.py --cfg_path configs/config.yaml
```

## Project Structure
```
.
├── code/
│   ├── datasets/       # Dataset loading and preprocessing
│   ├── models/         # Model architectures
│   ├── evaluate.py     # Evaluation script
│   └── utils.py        # Utility functions
├── configs/
│   └── config.yaml     # Configuration file
└── scripts/            # Training and evaluation scripts
```

## Note
`scripts/` provides example training and evaluation commands; `configs/config.yaml` is the main entry point for paths and model settings.

## Acknowledgments
This project builds upon the [DIF-Gaussian](https://github.com/xmed-lab/DIF-Gaussian) codebase.

