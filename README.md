# UrbanTwin LUMPI Track — 3rd Place Solution

**Challenge:** UCF UrbanTwin Sim2Real LiDAR Challenge (LUMPI Track)  
**Venue:** 6th DriveX Workshop @ ECCV 2026  
**Team:** Anonymous  
**Rank:** 3rd Place  
**Combined Score:** 0.4637 (Detection: 0.1690, Realism: 0.9057)

---

## Overview

This repository contains our complete solution for the LUMPI Track of the
UrbanTwin Sim2Real Challenge. The task is to generate synthetic LiDAR point
clouds and train a 3D object detector purely on synthetic data, then evaluate
on held-out real frames.

Our approach:
1. Generate synthetic roadside LiDAR in CARLA with 5 sensors matching real LUMPI config
2. Apply geometry rescaling to align synthetic object dimensions with real medians
3. Train 4 detector variants (PointPillar and SECOND)
4. Fuse with per-class weighted bounding-box fusion (WBF)
5. Apply domain-adaptive box calibration and point-count filtering
6. Generate realistic synthetic frames for the realism component

## Quick Start

### Requirements

```bash
# Python 3.10+
conda create -n urbantwin python=3.10
conda activate urbantwin
pip install -r requirements.txt
```

### Hardware
- GPU: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4090, RTX 3090)
- RAM: 32GB+ recommended
- Storage: ~20GB for checkpoints and data

### Reproduce Submission G

```bash
# 1. Set up data paths
export URBANTWIN_ROOT=/path/to/this/repo

# 2. Generate synthetic data (requires CARLA server running)
python repos/HiFi-DT-Utils/set_sensors_and_gen_data.py \
    --num-frames 20000 --save-dir ./data/lumpi_v2

# 3. Apply geometry rescaling (creates lumpi_v3)
python rescale.py

# 4. Train detectors (OpenPCDet)
cd repos/OpenPCDet/tools
python train.py --cfg_file cfgs/pointpillar_lumpi_v3.yaml --batch_size 4
python train.py --cfg_file cfgs/second_lumpi.yaml --batch_size 4
# ... etc for all 4 models

# 5. Build final submission
python build_g.py
# Output: submissions/subG/submission.zip
```

## Repository Structure

```
urbantwin-lumpi/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image for reproducibility
├── CHALLENGE_REPORT.pdf         # 4-8 page challenge report
├── data/
│   ├── starter_kit/             # Challenge starter kit (release by organizers)
│   ├── lumpi_v2/                # Synthetic training data (CARLA)
│   ├── lumpi_v3/                # Rescaled training data
│   └── lumpi_real/              # Public real train split (217 frames)
├── repos/
│   ├── OpenPCDet/               # 3D detection framework
│   ├── HiFi-DT-Utils/           # CARLA synthetic data generation
│   ├── SEED/                    # SEED detector (unused in final)
│   └── urbantwin-sim2real-drivex2026/  # Challenge website
├── submissions/
│   └── subG/                    # Final submission files
├── build_g.py                   # Main submission builder
├── fpd_affine.py                # FPD affine transform for realism
├── resample.py                  # Occupancy resampling for realism
├── a3_calib.py                  # Box calibration on public real split
├── rescale.py                   # Geometry rescaling
├── ens4_cfg.json                # Ensemble weights and calibration
└── scripts/
    ├── compute_realism.py       # Local realism scoring
    ├── package_submission.py    # Submission packaging
    └── inference_lumpi_test.py  # Inference script
```

## Model Checkpoints

Due to size, model weights are hosted on **Hugging Face Hub**:

| Model | Size | HF Link |
|---|---|---|
| PointPillar v1 | 127MB | [Coming Soon] |
| PointPillar v4 | 127MB | [Coming Soon] |
| SECOND v5 | 163MB | [Coming Soon] |
| PointPillar v3 | 127MB | [Coming Soon] |

Total: ~544MB for all 4 model checkpoints.

To download:
```bash
huggingface-cli download urbantwin/lumpi-checkpoints --local-dir ./checkpoints
```

## Results

| Component | Score |
|---|---|
| 3D mAP @ IoU 0.5 (R40) | 0.1183 |
| Realism (normalized) | 0.9057 |
| **Combined** | **0.4637** |

### Per-class AP

| Class | AP | n_gt (test) | n_pred |
|---|---|---|---|
| Person | 0.1546 | 512 | 7,641 |
| Car | 0.6112 | 1,258 | 5,964 |
| Bicycle | 0.0231 | 82 | 4,229 |
| Motorcycle | 0.0006 | 27 | 33,912 |
| Bus | 0.0000 | 25 | 459 |
| Truck | 0.0255 | 192 | 1,511 |
| Van | 0.0130 | 79 | 2,110 |

## Key Design Decisions

### 1. Geometry Rescaling (Table 2 in report)
The most impactful single improvement was rescaling synthetic object dimensions
to match real medians derived from the 217-frame public train split. This
improved single-model mAP from ~0.141 to ~0.150.

### 2. Box Calibration (A3)
Per-class box calibration fitted on the public real split provided the largest
gain (+0.042 mAP), correcting systematic size and height bias.

### 3. Ensemble Diversity
Four models with different architectures (PointPillar vs SECOND) and training
data (v1 vs v3) provide complementary errors that fusion can correct.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{anonymous2026urbantwin,
  title={Ensemble Detectors with Domain-Adaptive Post-Processing for Sim2Real LiDAR Object Detection},
  author={Anonymous},
  booktitle={6th DriveX Workshop @ ECCV 2026},
  year={2026}
}
```

## License

MIT License - see LICENSE file.

## Contact

For questions about this submission, contact the team via Codabench or the
challenge organizers at Muhammad.Shahbaz@ucf.edu.
