# Dataset Preparation

This folder contains dataset-specific preprocessing code and the expected layout for training/evaluation. Use `data/LUNA16` and `data/ToothFairy` as references for new datasets.

## Quick Start (New Dataset)
1) Put raw data under `data/<YourDataset>/raw/`.
```
data/<YourDataset>/
  raw/
    <case_001>
    <case_002>
```

2) Implement `data/<YourDataset>/dataset.py` (follow `data/ToothFairy/dataset.py` or `data/LUNA16/dataset.py`). The loader should return:
- `name`: case id
- `image`: 3D volume (numpy array)
- `spacing`: voxel spacing in mm

3) Configure preprocessing + projector in `data/<YourDataset>/config.yaml`:
- `dataset.spacing`, `dataset.resolution`, `dataset.value_range`, `dataset.block_size`
- `projector.nVoxel`, `projector.nDetector`, `projector.dVoxel`, `projector.dDetector`, `projector.DSO`, `projector.DSD`, and angles

4) Define splits in `data/<YourDataset>/splits.json`, then run preprocessing:
```bash
export PYTHONPATH=$(pwd)/data/:$PYTHONPATH
cd data/<YourDataset>
bash run.sh
```

After preprocessing, you should have:
```
data/<YourDataset>/
  processed/
    images/
    blocks/
    projections/
  meta_info.json
```

5) Point `configs/config.yaml` to the parent `data/` folder and train with `--dst_name <YourDataset>`.

## Splits
Splits live in `data/<Dataset>/splits.json` and are merged into `meta_info.json` during preprocessing. Example:
```json
{
  "train": ["case_001", "case_002"],
  "val": ["case_003"],
  "test": ["case_004"]
}
```

## Notes
- `projs/*.pkl` is generated during preprocessing. Each file is a pickle dict with keys: `projs` (uint8, `[K, W, H]`), `projs_max` (float), and `angles` (float `[K]`).
- `images/*` can be any SimpleITK-readable volume (e.g., `.mhd`, `.nii`, `.nii.gz`) used for validation/testing.
