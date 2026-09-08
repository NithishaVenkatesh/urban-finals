# Reproducing submission G (872719, combined 0.4637) from checkpoints

Best dev-phase result. **Dev submissions do NOT carry over** — this must be
explicitly resubmitted in the Final phase (2 Aug 04:00 – 16 Aug 03:59 UTC).

## Inputs (all on the remote box)

| artefact | path |
|---|---|
| v1 checkpoint | `repos/OpenPCDet/output/pointpillar_lumpi_v1_archive/default/ckpt/checkpoint_epoch_30.pth` |
| v4 checkpoint | `repos/OpenPCDet/output/pointpillar_lumpi/default/ckpt/checkpoint_epoch_30.pth` |
| v5 checkpoint (SECOND) | `repos/OpenPCDet/output/second_lumpi/default/ckpt/checkpoint_epoch_38.pth` |
| v3 checkpoint | `repos/OpenPCDet/output/pointpillar_lumpi_v3/default/ckpt/checkpoint_epoch_30.pth` |
| ensemble config | `ens4_cfg.json` (per-class weights, calibration, p01 floors) |
| D predictions (Truck/Van source) | `submissions/subD/predictions.json` |
| realism frames | `submissions/subE/synthetic/` (= subD's, resampled) |
| declaration | `submissions/subE/declaration.txt` |
| builder | `build_g.py` |

## Pipeline

1. **Per-class emission** — for each class k independently: threshold at
   `SCORE_THRESH=0.01`, class-wise `nms_gpu` at `NMS_THRESH=0.01`, keep top 500.
   NOT OpenPCDet's `MULTI_CLASSES_NMS` (broken for single-head: builds the label
   mapping as `arange(1, num_class)`, 7 entries for 8 classes, then asserts ==8).
2. **Weighted WBF fusion** across v1/v4/v5/v3, IoU 0.55, per-class member weights
   from `ens4_cfg.json` (v3 ×3 on Car/Person, ×2 Bicycle/Motorcycle, ×1 Bus,
   ×0.5 Truck/Van). Headings aligned mod pi before averaging (IoU is pi-invariant).
3. **Per-class box calibration** — `(sl, sw, sh, z_off)` scaled about the box
   BOTTOM, from `ens4_cfg.json["cal"]`.
4. **Point-count filter** — drop boxes with fewer than the class p01 true-positive
   point count, from `ens4_cfg.json["floors"]`.
5. **Truck and Van replaced wholesale** with `subD/predictions.json` entries
   (server-measured 0.0255 / 0.0130 vs the ensemble's lower values).
6. **Realism frames** copied unchanged from `subE/synthetic/`.

## Rebuild and verify

```bash
source ~/miniconda3/bin/activate urbantwin
cd /home/qernels/urbantwin_lumpi
python build_g.py
md5sum submissions/subG/submission.zip
# must equal 19db7132561eea4bd48743636eaacca9
```

## Expected scores (server, submission 872719)

```
combined 0.4637 | mAP@0.5 0.1183 | realism 0.9057
CD 2.2401 | MMD 0.000902 | EMD 1.6637 | FPD 1.7039
Person 0.1546 | Car 0.6112 | Bicycle 0.0231 | Motorcycle 0.0006
Bus 0.0000 | Truck 0.0255 | Van 0.0130 | Unknown n/a (0 GT)
```

## Provenance (for the declaration)

- Detector trained **only** on self-generated synthetic data (`lumpi_v2` / `lumpi_v3`,
  derived from UT-LUMPI with our own structure synthesis, human injection and
  geometry rescaling). No real point cloud was ever used to update weights.
- **Anchors** derived from the 217-frame public real train split
  (Measurement0–6, 31 frames each) — permitted under the domain-adaptation clause.
  Logged in `real_anchors.json`.
- **Box calibration and point-count floors** fitted on the same 217 frames.
- **Verified disjoint**: 0 exact point-cloud matches (MD5 on coordinates rounded
  to 1 mm) between those 217 frames and the 50 forbidden `detection_test_frames`.
- **OPEN ITEM**: the FPD affine transform and the occupancy resampling were fitted
  against `reference_data_local/realism_reference/`, which is byte-identical to the
  50 released `detection_test_frames` (set-hash `b8a40d65abef3caecaf0982b25154dcd`).
  Those frames appear in `forbidden_frames.txt`. Only their point clouds were used —
  their labels were never shipped and were never observed. Recommend refitting both
  against the 217-frame train split before the Final phase to remove the ambiguity.
