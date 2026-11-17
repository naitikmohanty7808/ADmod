"""Supplementary Section S2: exploratory (pre-freeze) analyses on the development cohorts, recomputed.

The previous revision's code and trained ADmod models were lost with its environment (protocol/DEVIATIONS.md).
The classical-model analyses below are recomputed exactly from the regenerated development cohorts with the
restored regional models (cross-region ridge on the previous revision's lambda grid, as in that revision).
The previous revision's ADmod results survive only as seed-level summaries in its README; they are printed
as recorded and are not recomputed.

    python -m admod.explore_s2 --out ../paper_access/supp
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .baselines import (RegionalCross2, RegionalLinear2, RegionalLinear2WM, RegionalRateOnly2, RegionalTrend,
                        copy_last, region_means)
from .confirm import boot_weights, compare, method_metrics, subject_stats, variance_budget
from .train import load

DEV = {"test": "test", "atyp": "stress_atypical", "fresh": "stress_long_gap_fresh"}
# Recorded in the previous revision's README (3 seeds, mean +/- SD of regional change R^2); not recomputed.
V2_ADMOD = {"60 epochs": {"test": (0.42, 0.04), "atyp": (-0.08, 0.15), "fresh": (0.41, 0.01)},
            "30 epochs": {"test": (0.35, 0.01), "atyp": (-0.07, 0.17), "fresh": (0.19, 0.09)}}
V2_RECORD = {"Regional linear (2 scans)": {"test": 0.20, "atyp": 0.01, "fresh": 0.17},
             "Cross-region ridge (2 scans)": {"test": 0.44, "atyp": -1.86, "fresh": 0.43},
             "Noise ceiling": {"test": 0.58, "atyp": 0.55, "fresh": 0.77}}
B = 2000


def classical(tr, va):
    return {"Regional trend (1 scan)": RegionalTrend().fit(tr), "Regional linear (2 scans)": RegionalLinear2().fit(tr),
            "Regional linear, rate only (2 scans)": RegionalRateOnly2().fit(tr),
            "Regional linear + WM (2 scans)": RegionalLinear2WM().fit(tr),
            "Cross-region ridge (2 scans)": RegionalCross2().fit(tr, va)}


def preds(fitted, co):
    out = {"Copy last scan": copy_last(co["baseline"])}
    out.update({k: m.predict(co) for k, m in fitted.items()})
    out["Noise ceiling"] = co["followup_clean"]
    out["True change + measured baseline"] = co["baseline"] + (co["followup_clean"] - co["baseline_clean"])
    return out


def score(co, P):
    W = boot_weights(len(co["dt"]), B)
    sts = {k: subject_stats(p, co) for k, p in P.items()}
    met, draws = {}, {}
    for k, s in sts.items():
        met[k], draws[k] = method_metrics([s], W)
    return met, draws, sts


def bias(co):
    lab = co["labels"]
    d = region_means(co["baseline"], lab) - region_means(co["baseline_clean"], lab)
    per_region = np.nanmean(d, 0)
    return {"min": float(per_region.min()), "max": float(per_region.max()), "mean": float(np.nanmean(d)),
            "sd_region_error": float(np.nanstd(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="../paper_access/supp")
    ap.add_argument("--results", default="results/explore_s2.json")
    ap.add_argument("--tex-only", action="store_true", help="rewrite the LaTeX from the saved results")
    a = ap.parse_args()
    if a.tex_only:
        write_tex(json.load(open(a.results)), a.out)
        return
    L = lambda n: load(os.path.join(a.data, n + ".npz"))
    tr, va = L("train"), L("val")
    fitted = classical(tr, va)
    res = {"ridge_lambda": fitted["Cross-region ridge (2 scans)"].lam, "sets": {}}
    for key, name in DEV.items():
        co = L(name)
        met, draws, sts = score(co, preds(fitted, co))
        r = {"n": int(len(co["dt"])), "methods": {k: {m: met[k][m] for m in ("regional_r2", "composite_mae")} for k in met},
             "variance_budget": variance_budget(co), "bias": bias(co)}
        r["ridge_vs_linear"] = compare(met["Cross-region ridge (2 scans)"], met["Regional linear (2 scans)"],
                                       draws["Cross-region ridge (2 scans)"], draws["Regional linear (2 scans)"],
                                       [sts["Cross-region ridge (2 scans)"]], [sts["Regional linear (2 scans)"]], sided="two")
        res["sets"][key] = r
    # Misregistration: refit at each shift amplitude (ridge lambda fixed at the main selection)
    lam = res["ridge_lambda"]
    res["misreg"] = {}
    for m in ("0.0", "0.1", "0.2", "0.3"):
        trm, tem = L(f"misreg_train_{m}"), L(f"misreg_test_{m}")
        fm = {"Regional linear (2 scans)": RegionalLinear2().fit(trm), "Cross-region ridge (2 scans)": RegionalCross2(lam=lam).fit(trm)}
        met, _, _ = score(tem, {k: v.predict(tem) for k, v in fm.items()})
        res["misreg"][m] = {"r2": {k: met[k]["regional_r2"] for k in met}, "variance_budget": variance_budget(tem),
                            "bias": bias(tem)}
    # Noise sweep: regional models refit at each noise level
    res["noise_sweep"] = {}
    for m in ("0.5", "1.0", "1.5", "2.0"):
        trm, tem = L(f"sweep_train_{m}"), L(f"sweep_test_{m}")
        fm = {"Regional trend (1 scan)": RegionalTrend().fit(trm), "Regional linear (2 scans)": RegionalLinear2().fit(trm),
              "Cross-region ridge (2 scans)": RegionalCross2(lam=lam).fit(trm)}
        P = {k: v.predict(tem) for k, v in fm.items()}
        P["Noise ceiling"] = tem["followup_clean"]
        P["True change + measured baseline"] = tem["baseline"] + (tem["followup_clean"] - tem["baseline_clean"])
        met, _, _ = score(tem, P)
        res["noise_sweep"][m] = {k: met[k]["regional_r2"] for k in met}
    res["v2_admod_recorded"] = V2_ADMOD
    res["v2_record_check"] = {k: {s: [v, res["sets"][s]["methods"][k]["regional_r2"]["mean"]] for s, v in d.items()}
                              for k, d in V2_RECORD.items()}
    os.makedirs(os.path.dirname(a.results), exist_ok=True)
    json.dump(res, open(a.results, "w"), indent=1)
    write_tex(res, a.out)
    print(json.dumps(res["v2_record_check"]))


# ------------------------------------------------------------------ LaTeX
SH = {"Copy last scan": "Copy", "Regional trend (1 scan)": "Trend", "Regional linear (2 scans)": "Linear",
      "Regional linear, rate only (2 scans)": "RateOnly", "Regional linear + WM (2 scans)": "LinearWM",
      "Cross-region ridge (2 scans)": "Ridge", "Noise ceiling": "Ceiling", "True change + measured baseline": "TrueAdd"}
SETN = {"test": "Test", "atyp": "Atyp", "fresh": "Fresh"}
MR = {"0.0": "Zero", "0.1": "One", "0.2": "Two", "0.3": "Three"}


def f3(x):
    return f"$-${abs(x):.3f}" if x < 0 else f"{x:.3f}"


def f2(x):
    return f"$-${abs(x):.2f}" if x < 0 else f"{x:.2f}"


def write_tex(res, out):
    M = []
    m = lambda k, v: M.append(rf"\newcommand{{\x{k}}}{{{v}}}")
    for s, S in SETN.items():
        r = res["sets"][s]
        m(f"N{S}", r["n"])
        for k, v in r["methods"].items():
            m(f"R{S}{SH[k]}", f3(v["regional_r2"]["mean"]))
        vb = r["variance_budget"]
        m(f"Vb{S}True", f"{100 * vb['true_change']:.0f}"); m(f"Vb{S}Base", f"{100 * vb['baseline_error']:.0f}")
        m(f"Vb{S}Follow", f"{100 * vb['followup_error']:.0f}")
        c = r["ridge_vs_linear"]
        m(f"D{S}RidgeLinear", f3(c["delta_regional_r2"])); m(f"Dlo{S}RidgeLinear", f3(c["ci95"][0])); m(f"Dhi{S}RidgeLinear", f3(c["ci95"][1]))
    b = res["sets"]["test"]["bias"]
    m("BiasMin", f"{b['min']:.3f}"); m("BiasMax", f"{b['max']:.3f}"); m("BiasMean", f"{b['mean']:.3f}")
    m("BiasMrZero", f"{res['misreg']['0.0']['bias']['mean']:.3f}")
    for k, d in res["misreg"].items():
        K = MR[k]
        m(f"MrSd{K}", f"{d['bias']['sd_region_error']:.3f}"); m(f"MrTrue{K}", f"{100 * d['variance_budget']['true_change']:.0f}")
        m(f"MrLinear{K}", f3(d["r2"]["Regional linear (2 scans)"]["mean"])); m(f"MrRidge{K}", f3(d["r2"]["Cross-region ridge (2 scans)"]["mean"]))
    lam = res["ridge_lambda"]
    m("Lambda", f"{lam:g}")
    open(os.path.join(out, "numbers_s2.tex"), "w").write("% generated by admod/explore_s2.py; do not edit\n" + "\n".join(M) + "\n")
    rows = ["Copy last scan", "Regional trend (1 scan)", "Regional linear (2 scans)", "Regional linear, rate only (2 scans)",
            "Regional linear + WM (2 scans)", "Cross-region ridge (2 scans)"]
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Exploratory: regional change $R^2$ (95\% subject-bootstrap interval) on the development cohorts of the previous revision. "
         r"Classical rows are recomputed from the regenerated cohorts. ADmod rows are the previous revision's recorded mean $\pm$ SD over 3 seeds; "
         r"its models were lost, so they have no interval and are not recomputed. Not confirmatory.}",
         r"\label{tab:s2main}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccc}", r"\toprule",
         "Method & " + " & ".join([f"Test ($n={res['sets']['test']['n']}$)", f"Atypical ($n={res['sets']['atyp']['n']}$)",
                                    f"6--8\\,y fresh ($n={res['sets']['fresh']['n']}$)"]) + r"\\", r"\midrule"]
    cell = lambda v: f"{f3(v['mean'])} ({f3(v['ci'][0])} to {f3(v['ci'][1])})"
    for k in rows:
        T.append(k + " & " + " & ".join(cell(res["sets"][s]["methods"][k]["regional_r2"]) for s in SETN) + r"\\")
    for ep, d in V2_ADMOD.items():
        T.append(f"ADmod, previous revision, {ep} (recorded) & " + " & ".join(f"{f2(d[s][0])} $\\pm$ {d[s][1]:.2f}" for s in SETN) + r"\\")
    T += [r"\midrule", r"\textit{Noise-free reference} & " + " & ".join(cell(res["sets"][s]["methods"]["Noise ceiling"]["regional_r2"]) for s in SETN) + r"\\",
          r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(out, "table_main.tex"), "w").write("\n".join(T) + "\n")
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Exploratory: effect of the random misregistration amplitude on the misregistration test cohorts. "
         r"SD of region-scale baseline error (SUVR), share of measured regional change that is true change, and regional change $R^2$ of models refit at each amplitude.}",
         r"\label{tab:misreg}", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Shift amplitude (voxels) & Baseline error SD & True-change share & Regional linear & Cross-region ridge\\", r"\midrule"]
    for k, d in res["misreg"].items():
        T.append(f"$\\pm${k} & {d['bias']['sd_region_error']:.3f} & {100 * d['variance_budget']['true_change']:.0f}\\% & "
                 f"{f3(d['r2']['Regional linear (2 scans)']['mean'])} & {f3(d['r2']['Cross-region ridge (2 scans)']['mean'])}\\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "table_misreg.tex"), "w").write("\n".join(T) + "\n")
    fig_noise(res, os.path.join(out, "fig_noise.pdf"))


def fig_noise(res, path):
    from .make_tex_ax import AQUA, BLUE, INK2, ORANGE, _style
    plt = _style()
    sw = res["noise_sweep"]
    ms = sorted(sw, key=float)
    xs = np.array([float(x) for x in ms])
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    series = [("Cross-region ridge (2 scans)", BLUE, "-", "o"), ("Regional linear (2 scans)", ORANGE, "-", "s"),
              ("Regional trend (1 scan)", AQUA, "-", "^"), ("Noise ceiling", INK2, "--", None),
              ("True change + measured baseline", INK2, ":", None)]
    label = {"Cross-region ridge (2 scans)": "Cross-region ridge", "Regional linear (2 scans)": "Regional linear",
             "Regional trend (1 scan)": "Regional trend", "Noise ceiling": "Noise-free reference",
             "True change + measured baseline": "True change + measured baseline"}
    for k, col, ls, mk in series:
        y = np.array([sw[x][k]["mean"] for x in ms])
        if mk:
            lo = np.array([sw[x][k]["ci"][0] for x in ms]); hi = np.array([sw[x][k]["ci"][1] for x in ms])
            ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
            ax.plot(xs, y, color=col, ls=ls, lw=1.6, marker=mk, ms=4, markeredgecolor="white", markeredgewidth=0.6, label=label[k])
        else:
            ax.plot(xs, y, color=col, ls=ls, lw=1.2, label=label[k])
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([f"×{x:g}" for x in xs])
    ax.set_xlabel("Scan noise relative to calibrated level")
    ax.set_ylabel("Regional change $R^2$")
    ax.legend(fontsize=6.5, loc="lower left", ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
