# DuFal: Dual-Frequency-Aware Learning for High-Fidelity Extremely Sparse-View CBCT Reconstruction

Cuong Tran, Trong-Thang Pham, Son Nguyen, Duy Minh Ho Nguyen, Ngan Le

## :tada: [Accepted at Transactions on Machine Learning Research (TMLR), December 2025](https://openreview.net/pdf?id=2wAZjAtK16)

[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg)](https://openreview.net/forum?id=2wAZjAtK16)

---

## Overview
Sparse-view **Cone-Beam Computed Tomography (CBCT)** reconstruction remains challenging due to severe undersampling of high-frequency anatomical details. Conventional CNN-based methods are often biased toward low-frequency information, leading to the loss of fine structures.

We propose **DuFal (Dual-Frequency-Aware Learning)**, a novel dual-path framework that jointly exploits spatial- and frequency-domain representations. At its core is a **High-Local Factorized Fourier Neural Operator**, which consists of:
- a **global high-frequency enhanced branch** for capturing long-range spectral patterns, and  
- a **local high-frequency enhanced branch** that processes spatial patches to preserve locality.

To improve efficiency, DuFal introduces **spectral–channel factorization** to reduce model complexity, along with a **cross-attention frequency fusion** module for effective integration of spatial and spectral features. The fused representations are decoded into projection features and reconstructed into 3D volumes via an **intensity field decoding** pipeline.

Experiments on **LUNA16** and **ToothFairy** demonstrate that DuFal consistently outperforms state-of-the-art methods, particularly under extremely sparse-view conditions, with superior preservation of high-frequency anatomical structures.


<p align="center">
    <img src="images/Xray2CT-TMLR-v2.png" width="90%"> <br>
  *Overview of DuFal architecture*
</p>

## 🏆 **Key Contributions**
- Dual-path encoding with spatial and frequency branches for high-frequency detail preservation.
- HiLocFFNO blocks combine global and local frequency modeling in a modular Frequency Encoder.
- Spectral-Channel Factorization (SCF) reduces FNO parameters while retaining quality.
- Cross-Attention Frequency Fusion (CAFF) merges spatial and spectral features in the frequency domain.
- Evaluated on LUNA16 and ToothFairy in extremely sparse-view settings.

---
## Table of Contents
- [📣 News](#-news)
- [🛠️ Requirements and Installation](#-requirements-and-installation)
- [📄 Configuration](#-configuration)
- [🧩 Creating splits for training and testing](#-creating-splits-for-training-and-testing)
- [🧪 Usage](#-usage)
- [🙏 Acknowledgments](#-acknowledgments)
- [📖 Citation](#-citation)
- [⚠️ Usage and License Notices](#-usage-and-license-notices)

---
## 📣 News
- **[Dec 2025]**: DuFal paper has been accepted with minor revisions at TMLR!

---
## 🛠️ Requirements and Installation
```bash
pip install torch==1.13 pytorch3d SimpleITK easydict
```

---
## 📄 Configuration
Edit `configs/config.yaml` to set your dataset path (after preprocessing):

```yaml
dataset:
  root_dir: /path/to/your/data
```

---
---
## 🧩 Creating splits for training and testing
Dataset preparation and split definitions are documented in `data/README.md`. Use the official splits for LUNA16 and ToothFairy (see `data/LUNA16/` and `data/ToothFairy/`).

---
## 🧪 Usage
### Training (single GPU)
```bash
CUDA_VISIBLE_DEVICES=0 python code/train.py \
  --batch_size 1 \
  --epoch 600 \
  --dst_name ToothFairy \
  --num_views 10 \
  --random_view \
  --cfg_path ./configs/config.yaml \
  --num_workers 8 \
  --eval_interval 50 \
  --save_interval 50 \
  --setting spatial-test \
  -trunc LL-LH LL-LH LL-LH LL-LH \
  -sobel 0 0 0 0 \
  -patch 1 1 1 1 \
  -psize 16 16 16 16 \
  -fac none none dep-sep dep-sep \
  -attn 0 0 0 1 \
  -prop 0 0 0 0 \
  -swin 0 \
  -wsize 0 0 0 0 \
  -grid linear \
  -fuse 8 \
  -fno \
  -skip add \
  --use_wandb
```


Key options:
- `--dst_name`: dataset name (`LUNA16` or `ToothFairy`).
- `--num_views`: number of projection views.
- `--cfg_path`: path to the config file.
- `--setting`: run name used for logging and outputs.
- `-trunc/-sobel/-patch/-psize/-fac/-attn/-prop/-swin/-wsize/-grid/-fuse/-fno/-skip`: architecture and frequency-encoding settings (kept consistent with the paper config).

### Testing
```bash
CUDA_VISIBLE_DEVICES=1 python code/evaluate.py --epoch xxx --dst_name {LUNA16 or ToothFairy} --split test --num_views 10 --out_res_scale 1.0 --setting xxx
```


---
## 🙏 Acknowledgments
This project builds upon the [DIF-Gaussian](https://github.com/xmed-lab/DIF-Gaussian) codebase.

---
## 📖 Citation
If you find this work useful, please cite our paper:

```bibtex
@article{tranvan2025dufal,
  title   = {DuFal: Dual-Frequency-Aware Learning for High-Fidelity Extremely Sparse-view CBCT Reconstruction},
  author  = {Tran Van, Cuong and Pham, Trong-Thang and Nguyen, Ngoc-Son and Ho Nguyen, Duy Minh and Le, Ngan},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  month   = {December}
}
```

---
## ⚠️ Usage and License Notices
The model is not intended for clinical use as a medical device, diagnostic tool, or any technology for disease diagnosis, treatment, or prevention. It is not a substitute for professional medical advice, diagnosis, or treatment. Users are responsible for evaluating and validating the model to ensure it meets their needs before any clinical application.
