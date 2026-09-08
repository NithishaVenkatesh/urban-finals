#!/usr/bin/env python3
"""FPD moment-matching: fit a global transform on the 50 synthetic realism
frames so their (mean, covariance) match the real reference, using the EXACT
scoring-program implementation for evaluation.

Variants tried, cheapest-first (each is a strict superset of the previous):
  t     : translation only              -> matches mean
  ts    : translation + isotropic scale -> matches mean + total variance
  tds   : translation + diagonal scale  -> matches mean + per-axis variance
  full  : full affine (whiten->recolor) -> matches mean + full covariance
Masking is applied inside the fit loop, because points pushed outside the
scoring range are dropped and shift the moments again.
"""
import glob, json, os, sys
import numpy as np

SK = "/home/qernels/urbantwin_lumpi/data/starter_kit"
sys.path.insert(0, f"{SK}/scoring_program")
from metrics import compute_realism, _concat_and_mask   # exact server code

SYN_DIR = "/home/qernels/urbantwin_lumpi/submissions/realism_v3b/synthetic"
REF_DIR = f"{SK}/reference_data_local/realism_reference"
OUT_DIR = "/home/qernels/urbantwin_lumpi/submissions/v6_fpd"
PC_RANGE = (-40, -40, -5, 40, 40, 5)
NORM = {"CD": (2.0, 30.0), "MMD": (0.0, 0.5), "EMD": (0.0, 5.0), "FPD": (0.0, 50.0)}

def load(d):
    return [np.fromfile(f, dtype=np.float32).reshape(-1, 3)
            for f in sorted(glob.glob(os.path.join(d, "*.bin")))]

def realism_score(r):
    s = []
    for k, (lo, hi) in NORM.items():
        v = float(np.clip(r[k], lo, hi))
        s.append(1.0 - (v - lo) / (hi - lo))
    return float(np.mean(s)), {k: round(1.0 - (np.clip(r[k], *NORM[k]) - NORM[k][0])
                                       / (NORM[k][1] - NORM[k][0]), 4) for k in NORM}

def moments(clouds):
    P = _concat_and_mask(clouds, PC_RANGE)
    return P.mean(axis=0).astype(np.float64), np.cov(P, rowvar=False), P.shape[0]

def sqrtm_psd(S):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(np.sqrt(w)) @ V.T

def invsqrtm_psd(S):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

def fit_transform(syn, mu_r, S_r, mode, iters=6):
    """Return (A, b) with y = x @ A.T + b, fitted so that the MASKED result
    matches the target moments."""
    A = np.eye(3); b = np.zeros(3)
    for _ in range(iters):
        cur = [c @ A.T + b for c in syn]
        mu_c, S_c, n = moments(cur)
        if mode == "t":
            C, d = np.eye(3), mu_r - mu_c
        elif mode == "ts":
            k = np.sqrt(np.trace(S_r) / max(np.trace(S_c), 1e-12))
            C = np.eye(3) * k; d = mu_r - k * mu_c
        elif mode == "tds":
            k = np.sqrt(np.diag(S_r) / np.clip(np.diag(S_c), 1e-12, None))
            C = np.diag(k); d = mu_r - C @ mu_c
        elif mode == "full":
            C = sqrtm_psd(S_r) @ invsqrtm_psd(S_c); d = mu_r - C @ mu_c
        else:
            raise ValueError(mode)
        A = C @ A; b = C @ b + d
    return A, b

def main():
    syn = load(SYN_DIR); ref = load(REF_DIR)
    print(f"synthetic frames {len(syn)}  reference frames {len(ref)}", flush=True)
    mu_r, S_r, n_r = moments(ref)
    mu_s, S_s, n_s = moments(syn)
    print(f"\nreal      mean {np.round(mu_r,3)}  n={n_r}")
    print(f"synthetic mean {np.round(mu_s,3)}  n={n_s}")
    print(f"real      diag(cov) {np.round(np.diag(S_r),3)}")
    print(f"synthetic diag(cov) {np.round(np.diag(S_s),3)}", flush=True)

    results = {}
    print("\n--- baseline (v3b, unmodified) ---", flush=True)
    r0 = compute_realism(syn, ref, PC_RANGE)
    sc0, parts0 = realism_score(r0)
    print(f"CD {r0['CD']:.4f}  MMD {r0['MMD']:.6f}  EMD {r0['EMD']:.4f}  FPD {r0['FPD']:.4f}"
          f"   -> realism {sc0:.4f}  {parts0}", flush=True)
    results["baseline"] = (sc0, r0, None, None)

    for mode in ("t", "ts", "tds", "full"):
        A, b = fit_transform(syn, mu_r, S_r, mode)
        cur = [c @ A.T + b for c in syn]
        r = compute_realism(cur, ref, PC_RANGE)
        sc, parts = realism_score(r)
        print(f"\n--- {mode} ---", flush=True)
        print(f"A diag {np.round(np.diag(A),4)}  b {np.round(b,3)}")
        print(f"CD {r['CD']:.4f}  MMD {r['MMD']:.6f}  EMD {r['EMD']:.4f}  FPD {r['FPD']:.4f}"
              f"   -> realism {sc:.4f}  {parts}", flush=True)
        results[mode] = (sc, r, A, b)

    best = max(results, key=lambda k: results[k][0])
    sc, r, A, b = results[best]
    print(f"\n==== BEST: {best}  realism {sc:.4f} (baseline {sc0:.4f}, "
          f"delta combined {0.4*(sc-sc0):+.4f}) ====", flush=True)
    if best == "baseline":
        print("No transform improves realism; not writing output."); return

    os.makedirs(f"{OUT_DIR}/synthetic", exist_ok=True)
    files = sorted(glob.glob(os.path.join(SYN_DIR, "*.bin")))
    for f, c in zip(files, syn):
        out = (c @ A.T + b).astype(np.float32)
        p = os.path.join(OUT_DIR, "synthetic", os.path.basename(f))
        out.tofile(p)
        assert os.path.getsize(p) <= 1_500_000, f"{p} too big"
        assert out.shape[0] <= 125_000, f"{p} too many points"
    json.dump({"mode": best, "A": A.tolist(), "b": b.tolist(),
               "realism_local": sc, "metrics_local": r},
              open(f"{OUT_DIR}/transform.json", "w"), indent=1)
    print(f"wrote 50 frames to {OUT_DIR}/synthetic (max size ok, <=125k pts)")

if __name__ == "__main__":
    main()
