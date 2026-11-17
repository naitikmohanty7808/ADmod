"""Confirmatory evaluation (IEEE Access revision). Implements protocol/protocol.json exactly.

Run once, after all models are trained and after this file is committed:

    python -m admod.dataset --confirm      # generates C1-C3 (seeds 21-23)
    python -m admod.confirm                # writes results/confirm.json and results/confirm_regional.npz
    python -m admod.confirm --sensitivity  # writes results/sensitivity.json (descriptive)

All data are SYNTHETIC.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from scipy.stats import wilcoxon

from .baselines import (RegionalCross2, RegionalLinear2, RegionalLinear2WM, RegionalRateOnly2, RegionalTrend,
                        copy_last)
from .phantom import N_REGIONS
from .train import AX_VARIANTS, VARIANTS, load, load_model, predict

PROTOCOL = "protocol/protocol.json"
FREEZE = "protocol/FREEZE_COMMIT.txt"
R = 2 * N_REGIONS
B_PRIMARY = 10_000
BOOT_SEED = 2027
COHORTS = {"C1": "confirm_id", "C2": "confirm_atypical", "C3": "confirm_long"}
RIDGE = "Cross-region ridge (2 scans)"
RLWM = "Regional linear + WM (2 scans)"
BASELINES = ["Copy last scan", "Regional trend (1 scan)", "Regional linear (2 scans)",
             "Regional linear, rate only (2 scans)", RLWM, RIDGE]
DEEP = {v: VARIANTS[v]["label"] for v in AX_VARIANTS}
MAIN = DEEP["ax_full"]
PLAIN = DEEP["ax_plain"]
CEILING = "Noise ceiling"
GAP_BINS = {"C1": [(1.0, 2.0), (2.0, 3.0), (3.0, 4.01)], "C3": [(6.0, 8.01)]}


# ------------------------------------------------------------------ per-subject sufficient statistics
def subject_stats(pred: np.ndarray, co: dict) -> dict:
    """Per-subject sums from which every metric (and its bootstrap) is computed exactly.

    Regional change R^2 = 1 - sum(SSE) / (sum(S2) - sum(S1)^2 / sum(N)), pooled over subjects and the
    16 bilateral cortical regions; this equals metrics.scores(...)["regional_r2"]."""
    lab, b, f, fc = co["labels"], co["baseline"], co["followup"], co["followup_clean"]
    n = len(pred)
    rp, rt, rc = np.full((n, R), np.nan), np.full((n, R), np.nan), np.full((n, R), np.nan)
    v_sse, v_s1, v_s2, v_n, comp = (np.zeros(n) for _ in range(5))
    for i in range(n):
        L = lab[i]
        m = (L >= 1) & (L <= R)
        li = L[m]
        cnt = np.bincount(li, minlength=R + 1)[1:].astype(float)
        dp = (pred[i] - b[i])[m].astype(np.float64)
        dt = (f[i] - b[i])[m].astype(np.float64)
        ok = cnt > 0
        rp[i, ok] = (np.bincount(li, weights=dp, minlength=R + 1)[1:] / np.maximum(cnt, 1))[ok]
        rt[i, ok] = (np.bincount(li, weights=dt, minlength=R + 1)[1:] / np.maximum(cnt, 1))[ok]
        pc = np.bincount(li, weights=pred[i][m].astype(np.float64), minlength=R + 1)[1:]
        tc = np.bincount(li, weights=fc[i][m].astype(np.float64), minlength=R + 1)[1:]
        rc[i, ok] = (pc / np.maximum(cnt, 1) - tc / np.maximum(cnt, 1))[ok]
        v_sse[i] = ((dp - dt) ** 2).sum(); v_s1[i] = dt.sum(); v_s2[i] = (dt ** 2).sum(); v_n[i] = m.sum()
        comp[i] = abs(float(pred[i][m].mean() - f[i][m].mean()))
    return {"rp": rp, "rt": rt, "r_sse": np.nansum((rp - rt) ** 2, 1), "r_s1": np.nansum(rt, 1),
            "r_s2": np.nansum(rt ** 2, 1), "r_n": np.sum(~np.isnan(rt), 1).astype(float),
            "r_mse": np.nanmean((rp - rt) ** 2, 1), "t_sse": np.nansum(rc ** 2, 1),
            "t_n": np.sum(~np.isnan(rc), 1).astype(float), "v_sse": v_sse, "v_s1": v_s1, "v_s2": v_s2,
            "v_n": v_n, "comp": comp}


def metric_rows(st: dict, W: np.ndarray) -> dict:
    """Metrics for each row of the weight matrix W (n_boot x n); a row of ones gives the point estimate."""
    def r2(sse, s1, s2, nn):
        return 1.0 - (W @ st[sse]) / (W @ st[s2] - (W @ st[s1]) ** 2 / (W @ st[nn]))
    w = W.sum(1)
    return {"regional_r2": r2("r_sse", "r_s1", "r_s2", "r_n"), "voxel_r2": r2("v_sse", "v_s1", "v_s2", "v_n"),
            "composite_mae": (W @ st["comp"]) / w, "truth_rmse": np.sqrt((W @ st["t_sse"]) / (W @ st["t_n"]))}


def boot_weights(n: int, B: int, seed: int = BOOT_SEED) -> np.ndarray:
    idx = np.random.default_rng(seed).integers(0, n, (B, n))
    W = np.zeros((B, n))
    np.add.at(W, (np.repeat(np.arange(B), n), idx.ravel()), 1.0)
    return W


def method_metrics(sts: list[dict], W: np.ndarray) -> tuple[dict, dict]:
    """Seed-averaged point estimates, per-seed values, 95% CIs; also returns the bootstrap draws."""
    one = np.ones((1, W.shape[1]))
    pts = [metric_rows(s, one) for s in sts]
    bs = [metric_rows(s, W) for s in sts]
    out, draws = {}, {}
    for k in pts[0]:
        per = np.array([p[k][0] for p in pts])
        d = np.mean([b[k] for b in bs], 0)
        draws[k] = d
        out[k] = {"mean": float(per.mean()), "sd": float(per.std(ddof=1)) if len(per) > 1 else 0.0,
                  "per_seed": per.tolist(), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out, draws


def subset_r2(sts: list[dict], sel: np.ndarray, B: int = 2000) -> dict:
    """Seed-averaged regional change R^2 on a subset of subjects, with a bootstrap inside the subset."""
    n = len(sts[0]["comp"])
    one = np.zeros((1, n)); one[0, sel] = 1.0
    Wb = np.zeros((B, n)); Wb[:, sel] = boot_weights(len(sel), B)
    pt = float(np.mean([metric_rows(s, one)["regional_r2"][0] for s in sts]))
    d = np.mean([metric_rows(s, Wb)["regional_r2"] for s in sts], 0)
    return {"mean": pt, "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def compare(a: dict, b: dict, da: dict, db: dict, sts_a, sts_b, delta0: float = 0.0, sided: str = "greater") -> dict:
    d_pt = a["regional_r2"]["mean"] - b["regional_r2"]["mean"]
    d = da["regional_r2"] - db["regional_r2"]
    B = len(d)
    if sided == "greater":
        p = (1 + np.sum(d <= delta0)) / (B + 1)
    else:
        p = min(1.0, 2 * min((1 + np.sum(d <= 0)) / (B + 1), (1 + np.sum(d >= 0)) / (B + 1)))
    ea = np.mean([s["r_mse"] for s in sts_a], 0)
    eb = np.mean([s["r_mse"] for s in sts_b], 0)
    return {"delta_regional_r2": float(d_pt), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "delta0": delta0, "p": float(p), "sided": sided,
            "delta_truth_rmse": a["truth_rmse"]["mean"] - b["truth_rmse"]["mean"],
            "delta_composite_mae": a["composite_mae"]["mean"] - b["composite_mae"]["mean"],
            "wilcoxon_p": float(wilcoxon(ea, eb).pvalue), "frac_subjects_better": float(np.mean(ea < eb))}


def holm(pvals: dict, alpha: float) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = {"p_holm": running, "reject": bool(running < alpha)}
    return out


# ------------------------------------------------------------------ models
def fit_baselines(tr: dict, va: dict) -> dict:
    return {"Regional trend (1 scan)": RegionalTrend().fit(tr), "Regional linear (2 scans)": RegionalLinear2().fit(tr),
            "Regional linear, rate only (2 scans)": RegionalRateOnly2().fit(tr), RLWM: RegionalLinear2WM().fit(tr),
            RIDGE: RegionalCross2(grid=RegionalCross2.LAMBDAS_EXT).fit(tr, va)}


def baseline_preds(fitted: dict, co: dict) -> dict:
    out = {"Copy last scan": [copy_last(co["baseline"])]}
    for k, m in fitted.items():
        out[k] = [m.predict(co)]
    return out


def load_deep(runs: str, expect_seeds=(0, 1, 2)) -> tuple[dict, dict]:
    models, info = {}, {}
    for v in AX_VARIANTS:
        paths = sorted(glob.glob(os.path.join(runs, f"{v}_s*", "best.pt")))
        loaded = [load_model(p) for p in paths]
        seeds = sorted(ck["seed"] for _, _, ck in loaded)
        assert seeds == list(expect_seeds), f"{v}: seeds {seeds}"
        for _, _, ck in loaded:
            assert ck["epochs"] == 100 and ck.get("bf16") and ck["prior"] is None, f"{v}: policy mismatch"
        models[v] = [m for m, _, _ in loaded]
        logs = [json.load(open(os.path.join(os.path.dirname(p), "train_log.json"))) for p in paths]
        info[v] = {"label": DEEP[v], "params": logs[0]["params"], "best_epoch": [l["best_epoch"] for l in logs],
                   "train_minutes": [l["minutes"] for l in logs], "best_val_loss": [l["best_val_loss"] for l in logs]}
    return models, info


def deep_preds(models: dict, co: dict) -> tuple[dict, dict]:
    out, secs = {}, {}
    for v, ms in models.items():
        t0 = time.time()
        out[DEEP[v]] = [predict(m, co, None) for m in ms]
        secs[DEEP[v]] = (time.time() - t0) / (len(ms) * len(co["dt"]))
    return out, secs


def check_cohort(co: dict, spec: dict, train_ids) -> None:
    assert bool(co["synthetic"]), "cohort not synthetic"
    assert len(co["dt"]) == spec["n"] and int(co["base_seed"]) == spec["generator_seed"], "cohort does not match protocol"
    assert str(co["pattern"]) == spec["pattern"]
    lo, hi = spec["forecast_gap_years"]
    assert co["dt"].min() >= lo - 1e-6 and co["dt"].max() <= hi + 1e-6
    assert not set(co["ids"]) & set(train_ids), "confirmatory subject in training data"


def variance_budget(co: dict) -> dict:
    from .baselines import region_means
    lab = co["labels"]
    rb, rf = region_means(co["baseline"], lab), region_means(co["followup"], lab)
    rbc, rfc = region_means(co["baseline_clean"], lab), region_means(co["followup_clean"], lab)
    v = np.nanvar((rf - rb).ravel())
    return {"true_change": float(np.nanvar((rfc - rbc).ravel()) / v), "baseline_error": float(np.nanvar((rb - rbc).ravel()) / v),
            "followup_error": float(np.nanvar((rf - rfc).ravel()) / v)}


# ------------------------------------------------------------------ confirmatory run
def run_confirm(data: str, runs: str, out: str) -> dict:
    proto = json.load(open(PROTOCOL))
    tr, va = load(os.path.join(data, "train.npz")), load(os.path.join(data, "val.npz"))
    train_ids = set(tr["ids"]) | set(va["ids"])
    fitted = fit_baselines(tr, va)
    models, train_info = load_deep(runs)
    torch.set_num_threads(2)

    res = {"protocol_freeze_commit": open(FREEZE).read().split()[0],
           "ridge_lambda": fitted[RIDGE].lam,
           "ridge_val_scores": {str(k): v for k, v in fitted[RIDGE].val_scores.items()},
           "training": train_info, "bootstrap": {"B": B_PRIMARY, "seed": BOOT_SEED}, "cohorts": {}}
    stats, draws, mets, regional = {}, {}, {}, {}
    for cid, fname in COHORTS.items():
        co = load(os.path.join(data, fname + ".npz"))
        check_cohort(co, proto["confirmatory_cohorts"][cid], train_ids)
        preds = baseline_preds(fitted, co)
        dp, secs = deep_preds(models, co)
        preds.update(dp)
        preds[CEILING] = [co["followup_clean"]]
        W = boot_weights(len(co["dt"]), B_PRIMARY)
        stats[cid] = {k: [subject_stats(p, co) for p in v] for k, v in preds.items()}
        mets[cid], draws[cid] = {}, {}
        for k, sts in stats[cid].items():
            mets[cid][k], draws[cid][k] = method_metrics(sts, W)
        bins = {}
        for lo, hi in GAP_BINS.get(cid, []):
            sel = np.where((co["dt"] >= lo) & (co["dt"] < hi))[0]
            bins[f"{lo:g}-{round(hi):g} y"] = {"n": int(len(sel)), **{
                k: subset_r2(stats[cid][k], sel) for k in (MAIN, PLAIN, RIDGE, RLWM, DEEP["ax_one_scan"], CEILING)}}
        regional[cid] = {k: np.stack([s["rp"] for s in v]) for k, v in stats[cid].items()}
        regional[cid]["_measured"] = stats[cid][CEILING][0]["rt"]
        res["cohorts"][cid] = {"file": fname + ".npz", "n": int(len(co["dt"])),
                               "forecast_gap": [float(co["dt"].min()), float(co["dt"].max())],
                               "methods": mets[cid], "inference_sec_per_subject": secs,
                               "variance_budget": variance_budget(co), "gap_bins": bins}

    hyp = proto["primary_endpoint"]["hypotheses"]
    prim = {}
    for h, spec in hyp.items():
        cid, comp = spec["cohort"], spec["comparator"]
        prim[h] = {"cohort": cid, "comparator": comp, "type": spec["type"],
                   **compare(mets[cid][MAIN], mets[cid][comp], draws[cid][MAIN], draws[cid][comp],
                             stats[cid][MAIN], stats[cid][comp], delta0=spec["delta0"], sided="greater")}
    adj = holm({h: v["p"] for h, v in prim.items()}, alpha=0.025)
    for h in prim:
        prim[h].update(adj[h])
    res["primary"] = prim

    abl = {}
    for v in AX_VARIANTS:
        if v == "ax_full":
            continue
        lab = DEEP[v]
        abl[lab] = compare(mets["C1"][MAIN], mets["C1"][lab], draws["C1"][MAIN], draws["C1"][lab],
                           stats["C1"][MAIN], stats["C1"][lab], sided="two")
    adj = holm({k: v["p"] for k, v in abl.items()}, alpha=0.05)
    for k in abl:
        abl[k].update(adj[k])
    res["ablation"] = abl

    sec = {}
    for cid in COHORTS:
        sec[cid] = {k: compare(mets[cid][MAIN], mets[cid][k], draws[cid][MAIN], draws[cid][k],
                               stats[cid][MAIN], stats[cid][k], sided="two")
                    for k in BASELINES + [PLAIN]}
    res["secondary_pairs"] = sec

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "confirm.json"), "w") as f:
        json.dump(res, f, indent=1)
    flat = {f"{cid}|{k}": v for cid, d in regional.items() for k, v in d.items()}
    np.savez_compressed(os.path.join(out, "confirm_regional.npz"), **flat)
    return res


# ------------------------------------------------------------------ sensitivity (descriptive)
SENS = ["noise_0.5", "noise_1.0", "noise_1.5", "noise_2.0", "misreg_0.0", "misreg_0.1", "misreg_0.2", "misreg_0.3",
        "rate_sd_low", "rate_sd_high", "susc_flat", "susc_steep"]


def run_sensitivity(data: str, runs: str, out: str) -> dict:
    torch.set_num_threads(2)
    res = {}
    for s in SENS:
        d = os.path.join(data, "sens", s)
        tr, va, te = (load(os.path.join(d, f"{k}.npz")) for k in ("train", "val", "test"))
        assert bool(te["synthetic"]) and not set(te["ids"]) & set(tr["ids"])
        fitted = fit_baselines(tr, va)
        preds = baseline_preds(fitted, te)
        m, _, ck = load_model(os.path.join(runs, s, "ax_full_s0", "best.pt"))
        assert ck["epochs"] == 100 and ck.get("bf16")
        preds[MAIN] = [predict(m, te, None)]
        preds[CEILING] = [te["followup_clean"]]
        W = boot_weights(len(te["dt"]), 2000)
        sts = {k: [subject_stats(p, te) for p in v] for k, v in preds.items()}
        mm, dd = {}, {}
        for k in sts:
            mm[k], dd[k] = method_metrics(sts[k], W)
        log = json.load(open(os.path.join(runs, s, "ax_full_s0", "train_log.json")))
        res[s] = {"n_train": int(len(tr["dt"])), "n_val": int(len(va["dt"])), "n_test": int(len(te["dt"])),
                  "ridge_lambda": fitted[RIDGE].lam, "best_epoch": log["best_epoch"], "methods": mm,
                  "vs": {k: compare(mm[MAIN], mm[k], dd[MAIN], dd[k], sts[MAIN], sts[k], sided="two")
                         for k in (RIDGE, RLWM, "Regional linear (2 scans)")},
                  "variance_budget": variance_budget(te)}
        print(s, {k: round(v["regional_r2"]["mean"], 3) for k, v in mm.items()}, flush=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "sensitivity.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs_ax")
    ap.add_argument("--sens-runs", default="runs_sens")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sensitivity", action="store_true")
    a = ap.parse_args()
    if a.sensitivity:
        run_sensitivity(a.data, a.sens_runs, a.out)
        return
    r = run_confirm(a.data, a.runs, a.out)
    for h, v in r["primary"].items():
        print(h, v["cohort"], "vs", v["comparator"], f"dR2={v['delta_regional_r2']:+.3f}",
              [round(x, 3) for x in v["ci95"]], f"p={v['p']:.4g} holm={v['p_holm']:.4g}", "REJECT H0" if v["reject"] else "")


if __name__ == "__main__":
    main()
