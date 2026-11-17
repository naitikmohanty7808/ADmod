"""Calibrate simulator scan-rescan variability and report accumulation rates (SYNTHETIC).

Restored for the IEEE Access revision (the original file was lost with the repository); the outputs
are checked against the values recorded by the previous version (composite scan-rescan difference
2.4193521740816855 % at global_sd 0.018, 400 subjects, seed 900).

Scan-rescan: two scans of the same amyloid state with independent noise, reported as the mean absolute
percent difference |test - retest| / mean(test, retest) * 100 of the cortical composite SUVR and of
regional SUVR.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .amyloid import TYPICAL, sample_params, simulate, susceptibility_map
from .phantom import cortical_labels, make_brain
from .pet import sample_tissue, scan


def test_retest(global_sd: float = 0.018, n_subjects: int = 400, seed: int = 900, max_shift: float = 0.3) -> dict:
    comp, reg = [], []
    for k in range(n_subjects):
        rng = np.random.default_rng(seed + k)
        br = make_brain(rng, 32)
        gm, lab = br["cortex_gm"], br["labels"]
        p = sample_params(rng)
        s = susceptibility_map(lab, gm, TYPICAL, rng)
        (c,) = simulate(gm, s, p["rho"], p["eps"], p["D"], [float(rng.uniform(3, 28))])
        t = sample_tissue(rng)
        a, _ = scan(lab, c, t, rng, 1.0, global_sd, max_shift_vox=max_shift)
        b, _ = scan(lab, c, t, rng, 1.0, global_sd, max_shift_vox=max_shift)
        ca, cb = a[gm].mean(), b[gm].mean()
        comp.append(abs(ca - cb) / ((ca + cb) / 2) * 100)
        for r in cortical_labels():
            m = lab == r
            if m.any():
                ra, rb = a[m].mean(), b[m].mean()
                reg.append(abs(ra - rb) / ((ra + rb) / 2) * 100)
    return {"global_sd": global_sd, "max_shift": max_shift, "n_subjects": n_subjects, "seed": seed,
            "composite_mean_abs_pct_diff": float(np.mean(comp)), "composite_sd": float(np.std(comp)),
            "regional_mean_abs_pct_diff": float(np.mean(reg))}


def accumulation(paths: list[str], width: float = 0.2, start: float = 1.0) -> dict:
    """Yearly change of the noise-free composite SUVR, binned by noise-free baseline composite SUVR."""
    cb, rate = [], []
    for p in paths:
        with np.load(p) as z:
            b, f, lab, dt = z["baseline_clean"], z["followup_clean"], z["labels"], z["dt"]
        for i in range(len(dt)):
            m = (lab[i] >= 1) & (lab[i] <= 16)
            cb.append(b[i][m].mean()); rate.append((f[i][m].mean() - b[i][m].mean()) / dt[i])
    cb, rate = np.array(cb), np.array(rate)
    bins, lo = [], start
    while lo < cb.max():
        sel = (cb >= lo) & (cb < lo + width)
        if sel.any():
            bins.append({"lo": round(lo, 2), "hi": round(lo + width, 2), "n": int(sel.sum()), "rate": float(rate[sel].mean())})
        lo += width
    return {"n_subjects": int(len(cb)), "bins": bins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/calibration.json")
    a = ap.parse_args()
    res = {"test_retest": [test_retest()],
           "accumulation": accumulation([os.path.join(a.data, f"{k}.npz") for k in ("train", "val", "test")])}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["test_retest"][0]), json.dumps(res["accumulation"]["bins"][:4]))


if __name__ == "__main__":
    main()
