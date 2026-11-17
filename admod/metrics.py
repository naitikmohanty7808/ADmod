"""Evaluation metrics, computed in cortical gray matter where amyloid builds up.

Change explained (R^2): share of the real scan-to-scan change a method gets right.
    1.0 = perfect, 0 = no better than guessing the average change, below 0 = worse than that.
    "Copy the last scan" predicts zero change, so it scores at or below 0.
Reported at three levels:
    voxel     every cortical voxel (noisiest)
    regional  average change in each of 16 cortical regions (standard in amyloid PET research)
composite_mae: error in whole-cortex average SUVR at follow-up (the summary number clinics use).
"""
from __future__ import annotations

import numpy as np

from .phantom import N_REGIONS, cortical_labels


def cortex_mask(labels: np.ndarray) -> np.ndarray:
    return (labels >= 1) & (labels <= 2 * N_REGIONS)


def _r2(pc: np.ndarray, tc: np.ndarray) -> float:
    sst = ((tc - tc.mean()) ** 2).sum()
    return float(1.0 - ((pc - tc) ** 2).sum() / sst)


def per_subject(pred, target, baseline, labels) -> list[dict]:
    out = []
    for i in range(len(pred)):
        m = cortex_mask(labels[i])
        pc = (pred[i] - baseline[i])[m].astype(np.float64)
        tc = (target[i] - baseline[i])[m].astype(np.float64)
        rp, rt = [], []
        for lab in cortical_labels():
            r = labels[i] == lab
            if r.any():
                rp.append(float((pred[i] - baseline[i])[r].mean()))
                rt.append(float((target[i] - baseline[i])[r].mean()))
        out.append({"pc": pc, "tc": tc, "rp": np.array(rp), "rt": np.array(rt),
                    "comp_err": float(pred[i][m].mean() - target[i][m].mean()),
                    "comp_pred_chg": float(pc.mean()), "comp_true_chg": float(tc.mean())})
    return out


def scores(st: list[dict], idx=None) -> dict:
    idx = range(len(st)) if idx is None else idx
    sel = [st[i] for i in idx]
    pc = np.concatenate([s["pc"] for s in sel]); tc = np.concatenate([s["tc"] for s in sel])
    rp = np.concatenate([s["rp"] for s in sel]); rt = np.concatenate([s["rt"] for s in sel])
    return {
        "voxel_r2": _r2(pc, tc),
        "regional_r2": _r2(rp, rt),
        "voxel_rmse": float(np.sqrt(((pc - tc) ** 2).mean())),
        "composite_mae": float(np.mean([abs(s["comp_err"]) for s in sel])),
    }


def bootstrap_indices(n: int, n_boot: int = 1000, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def summarize(pred, target, baseline, labels, boots=None) -> dict:
    st = per_subject(pred, target, baseline, labels)
    res = scores(st)
    if boots:
        bs = [scores(st, b) for b in boots]
        for k in list(res):
            v = np.array([b[k] for b in bs])
            res[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res["_stats"] = st
    return res


def paired_diff(st_a: list[dict], st_b: list[dict], boots, key: str) -> dict:
    """Bootstrap CI for score(a) - score(b), resampling the same subjects for both methods."""
    d = np.array([scores(st_a, b)[key] - scores(st_b, b)[key] for b in boots])
    return {"diff": scores(st_a)[key] - scores(st_b)[key],
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
