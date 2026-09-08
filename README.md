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

## Checkpoints

```bash
huggingface-cli download Nithiishaa/urbantwin-lumpi-checkpoints --local-dir ./checkpoints
```

## Configs

- `config/ens4_cfg.json` — ensemble weights, calibration params, point-count floors
- `config/rescale_ratios.json` — geometry rescaling ratios
- `config/real_anchors.json` — anchor sizes from the public real split

See `REPRODUCE_G.md` for exact reproduction steps.
