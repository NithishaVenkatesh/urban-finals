# UrbanTwin LUMPI Track — Solution

Sim2Real LiDAR pipeline: synthetic data generation, 3D detector ensemble, and
synthetic-frame realism post-processing.

## Setup

```bash
conda create -n urbantwin python=3.10
conda activate urbantwin
pip install -r requirements.txt
```

## Usage

```bash
# 1. Rescale synthetic object geometry to real medians
python rescale.py

# 2. Train detectors (OpenPCDet, synthetic data only)
cd repos/OpenPCDet/tools
python train.py --cfg_file cfgs/pointpillar_lumpi_v3.yaml --batch_size 4
python train.py --cfg_file cfgs/second_lumpi.yaml --batch_size 4

# 3. Fit box calibration on the public real split
python a3_calib.py

# 4. Realism post-processing
python fpd_affine.py
python resample.py

# 5. Build the submission zip
python build_g.py
# -> submissions/subG/submission.zip
```

## Training configs (reconstructed)

`repos/OpenPCDet/tools` is upstream [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) at commit `233f849`, the version recorded in every released checkpoint (`pcdet+0.6.0+233f849`), plus three LUMPI configs in `tools/cfgs/`:

| Config | Models | Data | Epochs |
|---|---|---|---|
| `pointpillar_lumpi_v3.yaml` | v3 PointPillar | `lumpi_v3` (rescaled) | 30 |
| `pointpillar_lumpi.yaml` | v1, v4 PointPillar | `lumpi_v1` | 30 |
| `second_lumpi.yaml` | v5 SECOND | `lumpi_v1` | 50 (epoch 38 released) |

The original config files were lost with the training machine and were reconstructed:

- **Architecture and voxel geometry:** each released checkpoint loads with `strict=True`.
- **Schedule:** `adam_onecycle`, LR 0.003, batch 4 and epoch counts were recovered from the optimizer state stored in the checkpoints.
- **Anchors:** from `config/real_anchors.json` (v3); v1/v4/v5 use synthetic sizes (real / `config/rescale_ratios.json`).
- **Assumed upstream defaults:** `WEIGHT_DECAY` and the augmentation list. These are not stored in the checkpoints.
- **Verification:** running the released checkpoints through the original `ens4.py` procedure on the 217 public real frames gives v3 0.1553 and 4-model ensemble 0.1841 mAP@0.5 (R40), against 0.1501 / 0.1684 in the report. Per-class anchor heights may differ slightly from the lost originals.

## Checkpoints

```bash
huggingface-cli download Nithiishaa/urbantwin-lumpi-checkpoints --local-dir ./checkpoints
```

## Configs

- `config/ens4_cfg.json` — ensemble weights, calibration params, point-count floors
- `config/rescale_ratios.json` — geometry rescaling ratios
- `config/real_anchors.json` — anchor sizes from the public real split

See `REPRODUCE_G.md` for exact reproduction steps.
