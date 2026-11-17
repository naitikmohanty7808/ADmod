"""Post hoc analyses (IEEE Access revision). NOT part of the frozen protocol; added after an internal
review of the methods and committed before the confirmatory cohorts were scored. All are descriptive.

    python -m admod.posthoc            # before scoring: checks only (cohort digests, checkpoints)
    python -m admod.posthoc --analyze  # after admod.confirm: writes results/posthoc.json

1. Integrity checks: content digests of the confirmatory cohorts against
   protocol/CONFIRM_COHORT_DIGESTS.txt; every checkpoint's architecture, scan count, width, epochs,
   lr, batch and precision against train.py.
2. Bias-corrected noise-free reference: the noise-free follow-up plus the per-region mean of
   (measured - noise-free) follow-up SUVR estimated on the training subjects.
3. All nine seed pairings for P2 (ADmod seed i minus plain 3D U-Net seed j) on C1.
4. Regional-change RMSE against the measured follow-up per forecast-gap bin.
5. Median seconds per training epoch per variant (contention-robust timing).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from .baselines import region_means
from .confirm import COHORTS, DEEP, GAP_BINS, MAIN, PLAIN
from .dataset import array_digest
from .train import AX_VARIANTS, VARIANTS, load


def check_digests(data: str) -> dict:
    out = {}
    for line in open("protocol/CONFIRM_COHORT_DIGESTS.txt").read().strip().splitlines():
        h, f = line.split()
        got = array_digest(os.path.join(data, os.path.basename(f)))
        assert got == h, f"content digest mismatch: {f}"
        out[os.path.basename(f)] = h
    return out


def check_checkpoints(runs: str) -> int:
    n = 0
    for v in AX_VARIANTS:
        for p in sorted(glob.glob(os.path.join(runs, f"{v}_s*", "best.pt"))):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            cfg = VARIANTS[v]
            assert ck["variant"] == v and ck["n_scans"] == cfg["n_scans"] and ck["arch"] == cfg["arch"], p
            assert ck["width"] == 12 and ck["epochs"] == 100 and ck["lr"] == 2e-3 and ck["batch"] == 4 and ck["bf16"], p
            n += 1
    return n


def _r2(rp, rt):
    m = ~np.isnan(rt)
    e = ((rp - rt)[m] ** 2).sum()
    t = rt[m]
    return 1.0 - e / ((t - t.mean()) ** 2).sum()


def analyze(data: str, runs: str, results: str) -> dict:
    Z = np.load(os.path.join(results, "confirm_regional.npz"))
    tr = load(os.path.join(data, "train.npz"))
    lab = tr["labels"]
    bias = np.nanmean(region_means(tr["followup"], lab) - region_means(tr["followup_clean"], lab), 0)
    res = {"note": "post hoc, descriptive; not part of the frozen protocol",
           "train_bias_per_region": bias.tolist(), "train_bias_mean": float(np.mean(bias))}
    ref = {}
    for cid in COHORTS:
        rt = Z[f"{cid}|_measured"]
        clean = Z[f"{cid}|Noise ceiling"][0]
        ref[cid] = {"noise_free_reference": float(_r2(clean, rt)), "bias_corrected_reference": float(_r2(clean + bias, rt)),
                    "admod": float(np.mean([_r2(p, rt) for p in Z[f"{cid}|{MAIN}"]]))}
    res["reference"] = ref
    rt = Z["C1|_measured"]
    A, P = Z[f"C1|{MAIN}"], Z[f"C1|{PLAIN}"]
    pair = [[float(_r2(A[i], rt) - _r2(P[j], rt)) for j in range(len(P))] for i in range(len(A))]
    res["p2_seed_pairings"] = {"delta": pair, "min": float(np.min(pair)), "max": float(np.max(pair)),
                               "n_positive": int(np.sum(np.array(pair) > 0))}
    bins = {}
    for cid, blist in GAP_BINS.items():
        dt = load(os.path.join(data, COHORTS[cid] + ".npz"))["dt"]
        rt = Z[f"{cid}|_measured"]
        for lo, hi in blist:
            sel = (dt >= lo) & (dt < hi)
            name = f"{lo:g}-{round(hi):g} y"
            bins[name] = {}
            for k in (MAIN, PLAIN, "Cross-region ridge (2 scans)", "Regional linear + WM (2 scans)", "Noise ceiling"):
                preds = Z[f"{cid}|{k}"]
                rm = [float(np.sqrt(np.nanmean((p[sel] - rt[sel]) ** 2))) for p in preds]
                bins[name][k] = float(np.mean(rm))
            bins[name]["sd_measured_change"] = float(np.nanstd(rt[sel]))
    res["gap_bin_rmse"] = bins
    timing = {}
    for v in AX_VARIANTS:
        secs = []
        for p in glob.glob(os.path.join(runs, f"{v}_s*", "train_log.json")):
            secs += [r["sec"] for r in json.load(open(p))["log"]]
        timing[DEEP[v]] = {"median_sec_per_epoch": float(np.median(secs)), "median_minutes_per_100_epochs": float(np.median(secs) * 100 / 60)}
    res["timing"] = timing
    with open(os.path.join(results, "posthoc.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs_ax")
    ap.add_argument("--results", default="results")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    print("cohort digests match:", list(check_digests(a.data)))
    print("checkpoints verified:", check_checkpoints(a.runs))
    if a.analyze:
        r = analyze(a.data, a.runs, a.results)
        print(json.dumps({k: r[k] for k in ("reference", "p2_seed_pairings")}, indent=1))


if __name__ == "__main__":
    main()
