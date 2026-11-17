"""Numbers, tables and figures for the IEEE Access manuscript, generated from result files only.

    python -m admod.make_tex_ax --out ../paper_access

Reads results/confirm.json, results/sensitivity.json and results/posthoc.json (optional),
results/calibration.json, runs_parity/{bf16,fp32}/ax_full_s1/train_log.json and protocol/*_COMMIT.txt. Every number in the
manuscript that comes from an experiment is a macro written here.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .make_tex import calib_macros

SHORT = {
    "Copy last scan": "Copy", "Regional trend (1 scan)": "Trend", "Regional linear (2 scans)": "Linear",
    "Regional linear, rate only (2 scans)": "RateOnly", "Regional linear + WM (2 scans)": "LinearWM",
    "Cross-region ridge (2 scans)": "Ridge", "ADmod": "Admod", "Plain 3D U-Net": "Plain",
    "ADmod without gap input": "NoGap", "ADmod without global context": "NoGlobal",
    "ADmod without region maps": "NoRegion", "ADmod with one scan": "OneScan",
    "ADmod with direct head": "Direct", "Noise ceiling": "Ceiling",
}
TABLE_NAME = {
    "Copy last scan": "Copy last scan", "Regional trend (1 scan)": "Regional trend (1 scan)",
    "Regional linear (2 scans)": "Regional linear", "Regional linear, rate only (2 scans)": "Regional linear, rate only",
    "Regional linear + WM (2 scans)": "Regional linear + WM", "Cross-region ridge (2 scans)": "Cross-region ridge",
    "ADmod": r"\textbf{ADmod}", "Plain 3D U-Net": "Plain 3D U-Net", "ADmod without gap input": "w/o gap input",
    "ADmod without global context": "w/o global context", "ADmod without region maps": "w/o region maps",
    "ADmod with one scan": "one scan", "ADmod with direct head": "direct head", "Noise ceiling": r"\textit{Noise-free reference}",
}
VARIANT_KEYS = {"ax_full": "ADmod", "ax_plain": "Plain 3D U-Net", "ax_no_gap": "ADmod without gap input",
                "ax_no_global": "ADmod without global context", "ax_no_region": "ADmod without region maps",
                "ax_one_scan": "ADmod with one scan", "ax_direct": "ADmod with direct head"}
REMOVED = {"ax_full": "(none)", "ax_plain": "global context, region maps, coordinates, two heads",
           "ax_no_gap": "gap channels and gap input to global MLP", "ax_no_global": "global context MLP",
           "ax_no_region": "region-mean maps", "ax_one_scan": "prior scan, its region map, $g_0$",
           "ax_direct": "two-head output (one head instead)"}
COH = {"C1": "COne", "C2": "CTwo", "C3": "CThree"}
HYP = {"P1": "One", "P2": "Two", "P3": "Three", "P4": "Four"}
BINS = {"1-2 y": "A", "2-3 y": "B", "3-4 y": "C", "6-8 y": "L"}


def f3(x):
    return f"{x:.3f}".replace("-", "$-$") if x < 0 else f"{x:.3f}"


def sgn3(x):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.3f}"


def pfmt(p):
    if p < 1e-4:
        return "$<$0.0001"
    if p < 1e-3:
        return f"{p:.4f}"
    return f"{p:.3f}"


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


class Macros:
    def __init__(self):
        self.lines = []

    def __call__(self, name, val):
        assert name.isalpha(), name
        self.lines.append(rf"\newcommand{{\{name}}}{{{val}}}")


def sens_name(s: str) -> str:
    S = "".join(w.capitalize() for w in s.replace(".", "p").replace("_", " ").split())
    S = "".join(ch for ch in S if ch.isalnum())
    for d, w in (("0", "Zero"), ("1", "One"), ("2", "Two"), ("3", "Three"), ("5", "Five")):
        S = S.replace(d, w)
    return S


def number_macros(res: dict, sens: dict | None, post: dict | None, cal: dict, parity: dict, commits: dict) -> list[str]:
    m = Macros()
    for cid, c in res["cohorts"].items():
        C = COH[cid]
        m(f"n{C}", c["n"])
        for k, v in c["methods"].items():
            s = SHORT[k]
            r = v["regional_r2"]
            m(f"r{C}{s}", f3(r["mean"])); m(f"rlo{C}{s}", f3(r["ci"][0])); m(f"rhi{C}{s}", f3(r["ci"][1]))
            if len(r["per_seed"]) > 1:
                m(f"sd{C}{s}", f"{r['sd']:.3f}")
                m(f"seeds{C}{s}", ", ".join(f3(x) for x in r["per_seed"]))
            m(f"mae{C}{s}", f"{v['composite_mae']['mean']:.4f}")
            m(f"trmse{C}{s}", f"{v['truth_rmse']['mean']:.4f}")
            m(f"vr{C}{s}", f3(v["voxel_r2"]["mean"]))
        vb = c["variance_budget"]
        m(f"vb{C}True", pct(vb["true_change"])); m(f"vb{C}Base", pct(vb["baseline_error"]))
        m(f"vb{C}Follow", pct(vb["followup_error"]))
        for b, d in c["gap_bins"].items():
            B = BINS[b]
            m(f"gn{B}", d["n"])
            for k, v in d.items():
                if k == "n":
                    continue
                s = SHORT[k]
                m(f"g{B}{s}", f3(v["mean"])); m(f"glo{B}{s}", f3(v["ci"][0])); m(f"ghi{B}{s}", f3(v["ci"][1]))
        if cid == "C1":
            for k, v in c["inference_sec_per_subject"].items():
                m(f"infer{SHORT[k]}", f"{1000 * v:.0f}")
    for h, v in res["primary"].items():
        H = HYP[h]
        m(f"p{H}D", sgn3(v["delta_regional_r2"])); m(f"p{H}Lo", f3(v["ci95"][0])); m(f"p{H}Hi", f3(v["ci95"][1]))
        m(f"p{H}P", pfmt(v["p"])); m(f"p{H}Holm", pfmt(v["p_holm"])); m(f"p{H}Wil", pfmt(v["wilcoxon_p"]))
        m(f"p{H}Better", pct(v["frac_subjects_better"]))
        m(f"p{H}Rej", "rejected" if v["reject"] else "not rejected")
        m(f"p{H}Mae", sgn3(v["delta_composite_mae"])); m(f"p{H}Trmse", sgn3(v["delta_truth_rmse"]))
    m("nPrimaryRejected", sum(v["reject"] for v in res["primary"].values()))
    for k, v in res["ablation"].items():
        s = SHORT[k]
        m(f"a{s}D", sgn3(v["delta_regional_r2"])); m(f"a{s}Lo", f3(v["ci95"][0])); m(f"a{s}Hi", f3(v["ci95"][1]))
        m(f"a{s}P", pfmt(v["p"])); m(f"a{s}Holm", pfmt(v["p_holm"])); m(f"a{s}Better", pct(v["frac_subjects_better"]))
        m(f"a{s}Rej", "significant" if v["reject"] else "not significant")
    for cid, d in res["secondary_pairs"].items():
        C = COH[cid]
        for k, v in d.items():
            s = SHORT[k]
            m(f"s{C}{s}D", sgn3(v["delta_regional_r2"])); m(f"s{C}{s}Lo", f3(v["ci95"][0]))
            m(f"s{C}{s}Hi", f3(v["ci95"][1])); m(f"s{C}{s}Wil", pfmt(v["wilcoxon_p"]))
            m(f"s{C}{s}Better", pct(v["frac_subjects_better"])); m(f"s{C}{s}Mae", sgn3(v["delta_composite_mae"]))
            m(f"s{C}{s}Trmse", sgn3(v["delta_truth_rmse"]))
    for v, info in res["training"].items():
        s = SHORT[VARIANT_KEYS[v]]
        m(f"params{s}", f"{info['params']:,}".replace(",", "{,}"))
        m(f"best{s}", ", ".join(str(e) for e in info["best_epoch"]))
    m("paramsFull", f"{res['training']['ax_full']['params']:,}".replace(",", "{,}"))
    allbest = [e for info in res["training"].values() for e in info["best_epoch"]]
    m("bestMin", min(allbest)); m("bestMax", max(allbest))
    m("bestNinety", sum(e >= 90 for e in allbest)); m("nModels", len(allbest))
    lam = res["ridge_lambda"]
    m("ridgeLambda", f"{lam:g}" if lam >= 0.01 else rf"10^{{{int(round(np.log10(lam)))}}}" if abs(np.log10(lam) - round(np.log10(lam))) < 1e-9 else f"{lam:g}")
    if parity:  # absent until the post hoc parity runs finish
        b, f = parity["bf16"], parity["fp32"]
        med = lambda d: float(np.median([r["sec"] for r in d["log"]]))
        m("bfSpeed", f"{med(f) / med(b):.1f}")
        m("bfValLoss", f"{b['best_val_loss']:.4f}"); m("fpValLoss", f"{f['best_val_loss']:.4f}")
        r2at = lambda d: [r for r in d["log"] if r["epoch"] == d["best_epoch"]][0]["val_regional_r2"]
        m("bfValRtwo", f"{r2at(b):.3f}"); m("fpValRtwo", f"{r2at(f):.3f}")
    for k, v in commits.items():
        m(k, v)
    m("codeAvailability", "The code is provided with this submission as supplementary material.")
    if sens:
        for s, d in sens.items():
            S = sens_name(s)
            for k in ("ADmod", "Cross-region ridge (2 scans)", "Regional linear + WM (2 scans)", "Noise ceiling"):
                m(f"sens{S}{SHORT[k]}", f3(d["methods"][k]["regional_r2"]["mean"]))
            m(f"sens{S}True", pct(d["variance_budget"]["true_change"]))
    if post:
        for cid, d in post["reference"].items():
            m(f"ref{COH[cid]}", f3(d["noise_free_reference"])); m(f"refBias{COH[cid]}", f3(d["bias_corrected_reference"]))
        m("trainBias", f"{post['train_bias_mean']:.3f}")
        pp = post["p2_seed_pairings"]
        m("pairMin", sgn3(pp["min"])); m("pairMax", sgn3(pp["max"])); m("pairPos", pp["n_positive"])
        for v, k in VARIANT_KEYS.items():
            m(f"min{SHORT[k]}", f"{post['timing'][k]['median_minutes_per_100_epochs']:.0f}")
        for b, d in post["gap_bin_rmse"].items():
            B = BINS[b]
            for k, val in d.items():
                if k == "sd_measured_change":
                    m(f"gsd{B}", f"{val:.4f}")
                else:
                    m(f"grm{B}{SHORT[k]}", f"{val:.4f}")
    return m.lines + calib_macros(cal)


# ------------------------------------------------------------------ tables
def table_primary(res: dict) -> list[str]:
    L = [r"\begin{table}[!t]", r"\caption{Primary hypotheses (frozen protocol). $\Delta$: regional change $R^2$ of ADmod minus comparator; 95\% paired subject-bootstrap interval; one-sided bootstrap $p$ ($B=10{,}000$), Holm-corrected over P1 to P4 at $\alpha=0.025$.}",
         r"\label{tab:primary}", r"\centering", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lllcc@{}}", r"\toprule",
         r"& Cohort, comparator & Test & $\Delta$ [95\% CI] & Holm $p$ \\", r"\midrule"]
    names = {"Regional linear + WM (2 scans)": "regional linear + WM", "Plain 3D U-Net": "plain 3D U-Net",
             "Cross-region ridge (2 scans)": "cross-region ridge"}
    for h, v in res["primary"].items():
        test = "superiority" if v["type"] == "superiority" else r"non-inf.\ ($-0.05$)"
        mark = r"$^{\ast}$" if v["reject"] else ""
        L.append(rf"{h} & {v['cohort']}, {names[v['comparator']]} & {test} & {sgn3(v['delta_regional_r2'])} [{f3(v['ci95'][0])}, {f3(v['ci95'][1])}] & {pfmt(v['p_holm'])}{mark} \\")
    L += [r"\bottomrule", r"\multicolumn{5}{@{}l}{\footnotesize $^{\ast}$Null hypothesis rejected.}", r"\end{tabular}}", r"\end{table}"]
    return L


def table_methods(res: dict) -> list[str]:
    c1, c2, c3 = (res["cohorts"][k]["methods"] for k in ("C1", "C2", "C3"))
    order = ["Copy last scan", "Regional trend (1 scan)", "Regional linear, rate only (2 scans)", "Regional linear (2 scans)",
             "Regional linear + WM (2 scans)", "Cross-region ridge (2 scans)", "Plain 3D U-Net", "ADmod", "Noise ceiling"]
    n = {k: res["cohorts"][k]["n"] for k in ("C1", "C2", "C3")}
    L = [r"\begin{table*}[!t]",
         r"\caption{All methods on the confirmatory cohorts. Regional change $R^2$ against the measured follow-up with 95\% subject-bootstrap intervals; composite MAE (SUVR) against the measured follow-up; regional RMSE (SUVR) against the noise-free follow-up. Deep models: mean over three seeds. Ablations are in Table~\ref{tab:ablation}.}",
         r"\label{tab:methods}", r"\centering", r"\setlength{\tabcolsep}{3.4pt}", r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         rf"& \multicolumn{{3}}{{c}}{{C1: in distribution, 1--4 y ($n={n['C1']}$)}} & \multicolumn{{2}}{{c}}{{C2: atypical pattern ($n={n['C2']}$)}} & \multicolumn{{2}}{{c}}{{C3: 6--8 y ($n={n['C3']}$)}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Method & $R^2$ [95\% CI] & MAE & RMSE$_\text{true}$ & $R^2$ [95\% CI] & RMSE$_\text{true}$ & $R^2$ [95\% CI] & RMSE$_\text{true}$ \\", r"\midrule"]
    for k in order:
        if k == "Noise ceiling":
            L.append(r"\midrule")
        row = [TABLE_NAME[k]]
        for c, extra in ((c1, ("composite_mae", "truth_rmse")), (c2, ("truth_rmse",)), (c3, ("truth_rmse",))):
            r = c[k]["regional_r2"]
            row.append(f"{f3(r['mean'])} [{f3(r['ci'][0])}, {f3(r['ci'][1])}]")
            for e in extra:
                row.append("--" if (k == "Noise ceiling" and e == "truth_rmse") else f"{c[k][e]['mean']:.4f}")
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return L


def table_ablation(res: dict) -> list[str]:
    c1 = res["cohorts"]["C1"]["methods"]
    L = [r"\begin{table}[!t]",
         r"\caption{Ablations on C1. $R^2$: mean (SD) over three seeds. $\Delta$: ADmod minus variant with 95\% paired bootstrap interval; two-sided $p$ Holm-corrected over six comparisons.}",
         r"\label{tab:ablation}", r"\centering", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Variant & $R^2$ (SD) & $\Delta$ [95\% CI] & Holm $p$ \\", r"\midrule"]
    r = c1["ADmod"]["regional_r2"]
    L.append(rf"\textbf{{ADmod}} & {f3(r['mean'])} ({r['sd']:.3f}) & -- & -- \\")
    for k, v in res["ablation"].items():
        r = c1[k]["regional_r2"]
        L.append(rf"{TABLE_NAME[k]} & {f3(r['mean'])} ({r['sd']:.3f}) & {sgn3(v['delta_regional_r2'])} [{f3(v['ci95'][0])}, {f3(v['ci95'][1])}] & {pfmt(v['p_holm'])} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


def table_variants(res: dict) -> list[str]:
    L = [r"\begin{table}[!t]", r"\caption{Deep model variants. Each ablation removes one component of ADmod. All were trained with seeds 0, 1, 2 under the same policy; best epoch is the epoch of lowest validation loss out of 100.}",
         r"\label{tab:variants}", r"\centering", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}p{0.27\columnwidth}p{0.34\columnwidth}rc@{}}", r"\toprule",
         r"Variant & Removed & Params & Best epochs \\", r"\midrule"]
    for v, info in res["training"].items():
        k = VARIANT_KEYS[v]
        name = r"\textbf{ADmod}" if v == "ax_full" else ("Plain 3D U-Net" if v == "ax_plain" else "ADmod " + TABLE_NAME[k])
        L.append(rf"{name} & {REMOVED[v]} & {info['params']:,} & {', '.join(str(e) for e in info['best_epoch'])} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


SENS_LABEL = {"noise_0.5": r"Noise $\times0.5$", "noise_1.0": r"Noise $\times1$", "noise_1.5": r"Noise $\times1.5$",
              "noise_2.0": r"Noise $\times2$", "misreg_0.0": r"Shift $\pm0$", "misreg_0.1": r"Shift $\pm0.1$",
              "misreg_0.2": r"Shift $\pm0.2$", "misreg_0.3": r"Shift $\pm0.3$", "rate_sd_low": r"Rate SD 0.10",
              "rate_sd_high": r"Rate SD 0.45", "susc_flat": r"Susceptibility$^{0.5}$", "susc_steep": r"Susceptibility$^{2}$"}


def table_sens(sens: dict) -> list[str]:
    L = [r"\begin{table}[!t]",
         r"\caption{Sensitivity analysis (descriptive). For each setting ADmod is retrained (seed 0) and the baselines are refit on 210 training subjects; test $n=120$. True: share of measured regional-change variance that is true change. $\Delta$: ADmod minus cross-region ridge, 95\% bootstrap interval.}",
         r"\label{tab:sens}", r"\centering", r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Setting & True & ADmod & Ridge & RL+WM & Ref. & $\Delta$ vs.\ ridge \\", r"\midrule"]
    for s, d in sens.items():
        mm = d["methods"]
        v = d["vs"]["Cross-region ridge (2 scans)"]
        L.append(rf"{SENS_LABEL[s]} & {pct(d['variance_budget']['true_change'])}\% & {f3(mm['ADmod']['regional_r2']['mean'])} & "
                 rf"{f3(mm['Cross-region ridge (2 scans)']['regional_r2']['mean'])} & {f3(mm['Regional linear + WM (2 scans)']['regional_r2']['mean'])} & "
                 rf"{f3(mm['Noise ceiling']['regional_r2']['mean'])} & {sgn3(v['delta_regional_r2'])} [{f3(v['ci95'][0])}, {f3(v['ci95'][1])}] \\")
        if s in ("noise_2.0", "misreg_0.3", "rate_sd_high"):
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


# ------------------------------------------------------------------ figures
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                         "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "pdf.fonttype": 42})
    return plt


def fig_forest(res: dict, path: str):
    plt = _style()
    prim = res["primary"]; abl = res["ablation"]
    rows = [(f"{h}: vs {'RL+WM' if 'WM' in v['comparator'] else ('plain U-Net' if 'Plain' in v['comparator'] else 'ridge')} ({v['cohort']})",
             v["delta_regional_r2"], v["ci95"], v["reject"]) for h, v in prim.items()]
    arows = [(f"vs {TABLE_NAME[k].replace('w/o', 'without')}" if k != "Plain 3D U-Net" else "vs plain 3D U-Net",
              v["delta_regional_r2"], v["ci95"], v["reject"]) for k, v in abl.items()]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.3), gridspec_kw={"height_ratios": [len(rows), len(arows)]})
    for ax, rr, title in ((axes[0], rows, "(a) Primary hypotheses"), (axes[1], arows, "(b) Ablations on C1: ADmod minus variant")):
        lo_all = min(r[2][0] for r in rr)
        hi_clip = 0.35
        xmin = min(-0.12, lo_all - 0.02)
        for i, (lab, d, ci, rej) in enumerate(rr):
            y = len(rr) - 1 - i
            off = ci[1] > hi_clip
            x_hi = min(ci[1], hi_clip); x_d = min(d, hi_clip)
            ax.plot([ci[0], x_hi], [y, y], color=INK, lw=1.2, solid_capstyle="round")
            ax.plot([x_d], [y], marker="o", ms=4.5, color=BLUE if rej else "white", mec=BLUE, mew=1.2, zorder=3)
            if off:
                ax.annotate(f"{d:+.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", xy=(hi_clip, y), xytext=(hi_clip - 0.01, y + 0.33),
                            ha="right", va="center", fontsize=6.5, color=INK2)
                ax.plot([hi_clip], [y], marker=">", ms=4, color=INK)
        ax.axvline(0, color=INK2, lw=0.8)
        if title.startswith("(a)"):
            ax.axvline(-0.05, color=INK2, lw=0.8, ls=(0, (3, 2)))
            ax.text(-0.056, len(rr) - 1.5, "NI\nmargin", fontsize=6.5, color=INK2, ha="right", va="center")
        ax.set_yticks(range(len(rr))); ax.set_yticklabels([r[0] for r in rr][::-1], fontsize=7)
        ax.set_xlim(xmin, hi_clip + 0.01); ax.set_ylim(-0.6, len(rr) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=8, loc="left", color=INK)
    axes[1].set_xlabel(r"$\Delta$ regional change $R^2$ (95% CI)")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_gap(res: dict, path: str):
    plt = _style()
    series = [("Noise ceiling", INK2, "--", "s"), ("ADmod", BLUE, "-", "o"), ("Cross-region ridge (2 scans)", ORANGE, "-", "D"),
              ("Regional linear + WM (2 scans)", AQUA, "-", "^")]
    labels = {"Noise ceiling": "noise-free reference", "ADmod": "ADmod", "Cross-region ridge (2 scans)": "cross-region ridge",
              "Regional linear + WM (2 scans)": "regional linear + WM"}
    b1 = res["cohorts"]["C1"]["gap_bins"]; b3 = res["cohorts"]["C3"]["gap_bins"]
    bins = [("1-2 y", b1), ("2-3 y", b1), ("3-4 y", b1), ("6-8 y", b3)]
    xs = [0, 1, 2, 3.6]
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    for j, (k, col, ls, mk) in enumerate(series):
        dx = (j - 1.5) * 0.07
        y = [b[name][k]["mean"] for name, b in bins]
        lo = [b[name][k]["ci"][0] for name, b in bins]; hi = [b[name][k]["ci"][1] for name, b in bins]
        x = [v + dx for v in xs]
        ax.plot(x[:3], y[:3], color=col, ls=ls, lw=1.2)
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt=mk, ms=4, color=col, mfc=col if k != "Noise ceiling" else "white",
                    elinewidth=0.9, capsize=0, label=labels[k], zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels([f"{n.replace('-', chr(8211))}\n(n={b[n]['n']})" for n, b in bins], fontsize=7)
    ax.axvline(3.0, color=GRID, lw=0.8)
    ax.text(1.0, ax.get_ylim()[0], "C1", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.text(3.6, ax.get_ylim()[0], "C3", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xlabel("Forecast gap (years)"); ax.set_ylabel(r"Regional change $R^2$")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.22), handlelength=2.2, columnspacing=1.0)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_sens(sens: dict, path: str):
    plt = _style()
    series = [("Noise ceiling", INK2, "--", "s", "noise-free reference"), ("ADmod", BLUE, "-", "o", "ADmod"),
              ("Cross-region ridge (2 scans)", ORANGE, "-", "D", "cross-region ridge"),
              ("Regional linear + WM (2 scans)", AQUA, "-", "^", "regional linear + WM")]
    panels = [("(a) Scan noise", ["noise_0.5", "noise_1.0", "noise_1.5", "noise_2.0"], ["0.5", "1", "1.5", "2"], r"noise multiplier $m$"),
              ("(b) Misregistration", ["misreg_0.0", "misreg_0.1", "misreg_0.2", "misreg_0.3"], ["0", "0.1", "0.2", "0.3"], "shift range (voxel)"),
              ("(c) Rate variability", ["rate_sd_low", "noise_1.0", "rate_sd_high"], ["0.10", "0.25", "0.45"], r"log-SD of $\rho$"),
              ("(d) Susceptibility gradient", ["susc_flat", "noise_1.0", "susc_steep"], ["0.5", "1", "2"], r"exponent $\gamma$")]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0), sharey=True)
    for ax, (title, keys, ticks, xl) in zip(axes, panels):
        x = np.arange(len(keys))
        for k, col, ls, mk, lab in series:
            y = [sens[s]["methods"][k]["regional_r2"]["mean"] for s in keys]
            lo = [sens[s]["methods"][k]["regional_r2"]["ci"][0] for s in keys]
            hi = [sens[s]["methods"][k]["regional_r2"]["ci"][1] for s in keys]
            ax.plot(x, y, color=col, ls=ls, lw=1.2)
            ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt=mk, ms=3.5, color=col,
                        mfc=col if k != "Noise ceiling" else "white", elinewidth=0.8, capsize=0, label=lab, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(ticks); ax.set_xlabel(xl, fontsize=7)
        ax.set_title(title, fontsize=8, loc="left"); ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.3, len(keys) - 0.7)
    axes[0].set_ylabel(r"Regional change $R^2$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=7)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ supplement
def supp_tables(res: dict, runs_dir: str, post: dict | None, out: str) -> list[str]:
    L = []
    coh = ("C1", "C2", "C3")
    L += [r"\begin{table}[h]", r"\centering",
          r"\caption{Regional change $R^2$ of every deep model, per seed (0, 1, 2), on each confirmatory cohort.}",
          r"\label{tab:seeds}", r"\small", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
          r"Variant & C1 & C2 & C3 \\", r"\midrule"]
    for v, k in VARIANT_KEYS.items():
        cells = [" / ".join(f3(x) for x in res["cohorts"][c]["methods"][k]["regional_r2"]["per_seed"]) for c in coh]
        L.append(k + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    if post:
        pp = post["p2_seed_pairings"]["delta"]
        L += [r"\begin{table}[h]", r"\centering",
              r"\caption{Post hoc: regional change $R^2$ on C1 of each ADmod seed minus each plain 3D U-Net seed.}",
              r"\label{tab:pairings}", r"\small", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
              r"& plain seed 0 & plain seed 1 & plain seed 2 \\", r"\midrule"]
        for i, row in enumerate(pp):
            L.append(f"ADmod seed {i} & " + " & ".join(sgn3(x) for x in row) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    L += [r"\begin{table}[h]", r"\centering",
          r"\caption{ADmod minus each comparator on each confirmatory cohort (secondary, unadjusted). $\Delta R^2$ with 95\% paired bootstrap interval; Wilcoxon signed-rank $p$ on per-subject regional squared error; share of subjects for whom ADmod has the lower error.}",
          r"\label{tab:pairs}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
          r"Cohort & Comparator & $\Delta R^2$ [95\% CI] & Wilcoxon $p$ & ADmod better \\", r"\midrule"]
    for c in coh:
        for k, v in res["secondary_pairs"][c].items():
            L.append(rf"{c} & {TABLE_NAME[k]} & {sgn3(v['delta_regional_r2'])} [{f3(v['ci95'][0])}, {f3(v['ci95'][1])}] & {pfmt(v['wilcoxon_p'])} & {pct(v['frac_subjects_better'])}\% \\")
        if c != "C3":
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    L += [r"\begin{table}[h]", r"\centering",
          r"\caption{Primary comparisons: secondary statistics. Differences are ADmod minus comparator.}",
          r"\label{tab:primsec}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r"& Raw one-sided $p$ & Wilcoxon $p$ & $\Delta$ composite MAE & $\Delta$ RMSE$_\text{true}$ \\", r"\midrule"]
    for h, v in res["primary"].items():
        L.append(rf"{h} & {pfmt(v['p'])} & {pfmt(v['wilcoxon_p'])} & {sgn3(v['delta_composite_mae'])} & {sgn3(v['delta_truth_rmse'])} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    bins = [("1-2 y", "C1"), ("2-3 y", "C1"), ("3-4 y", "C1"), ("6-8 y", "C3")]
    L += [r"\begin{table}[h]", r"\centering",
          r"\caption{Regional change $R^2$ by forecast gap with 95\% bootstrap intervals within each bin; in parentheses, regional-change RMSE (SUVR) against the measured follow-up (post hoc). Gaps of 1--4\,y come from C1, 6--8\,y from C3.}",
          r"\label{tab:gap}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          "Method & " + " & ".join(f"{b.replace('-', '--')} ($n={res['cohorts'][c]['gap_bins'][b]['n']}$)" for b, c in bins) + r" \\", r"\midrule"]
    for k in ("ADmod", "Plain 3D U-Net", "ADmod with one scan", "Cross-region ridge (2 scans)", "Regional linear + WM (2 scans)", "Noise ceiling"):
        cells = []
        for b, c in bins:
            d = res["cohorts"][c]["gap_bins"][b][k]
            cell = f"{f3(d['mean'])} [{f3(d['ci'][0])}, {f3(d['ci'][1])}]"
            if post and k in post["gap_bin_rmse"][b]:
                cell += f" ({post['gap_bin_rmse'][b][k]:.4f})"
            cells.append(cell)
        L.append(TABLE_NAME[k] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    L += [r"\begin{table}[h]", r"\centering",
          r"\caption{Training summary (100 epochs each, two CPU cores, bfloat16 autocast) and float32 inference time per subject on C1. Minutes per model: median epoch time $\times$ 100 (post hoc), which is robust to periods when other jobs shared the CPU.}",
          r"\label{tab:train}", r"\small", r"\begin{tabular}{@{}lrccc@{}}", r"\toprule",
          r"Variant & Params & Best epochs & Minutes per model & Inference (ms/subject) \\", r"\midrule"]
    inf = res["cohorts"]["C1"]["inference_sec_per_subject"]
    for v, info in res["training"].items():
        k = VARIANT_KEYS[v]
        mins = post["timing"][k]["median_minutes_per_100_epochs"] if post else np.mean(info["train_minutes"])
        L.append(rf"{k} & {info['params']:,} & {', '.join(map(str, info['best_epoch']))} & {mins:.0f} & {1000 * inf[k]:.0f} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    vs = res["ridge_val_scores"]
    L += [r"\begin{table}[h]", r"\centering",
          r"\caption{Cross-region ridge: validation regional change $R^2$ for each penalty $\lambda$ (extended grid). Bold: selected.}",
          r"\label{tab:lambda}", r"\small", r"\begin{tabular}{@{}rc@{\hspace{14pt}}rc@{}}", r"\toprule",
          r"$\lambda$ & val.\ $R^2$ & $\lambda$ & val.\ $R^2$ \\", r"\midrule"]
    items = list(vs.items()); half = (len(items) + 1) // 2

    def cell(kv):
        k, v = kv
        lab = f"{float(k):g}"
        val = f"{v:.4f}".replace("-", "$-$")
        return rf"\textbf{{{lab}}} & \textbf{{{val}}}" if float(k) == res["ridge_lambda"] else f"{lab} & {val}"
    for i in range(half):
        right = cell(items[i + half]) if i + half < len(items) else "&"
        L.append(cell(items[i]) + " & " + right + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    if post:
        L += [r"\begin{table}[h]", r"\centering",
              r"\caption{Post hoc: noise-free reference with and without the per-region measurement bias estimated on training subjects, and ADmod, on each confirmatory cohort (regional change $R^2$).}",
              r"\label{tab:reference}", r"\small", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
              r"& C1 & C2 & C3 \\", r"\midrule"]
        for key, lab in (("noise_free_reference", "Noise-free reference"), ("bias_corrected_reference", "Reference + training bias"), ("admod", "ADmod")):
            L.append(lab + " & " + " & ".join(f3(post["reference"][c][key]) for c in coh) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    fig_curves(res, runs_dir, os.path.join(out, "fig_curves.pdf"))
    L += [r"\begin{figure}[h]", r"\centering", r"\includegraphics[width=\textwidth]{fig_curves.pdf}",
          r"\caption{Validation regional change $R^2$ by epoch for the 21 models (three seeds per variant). Dots: epoch of lowest validation loss (the saved checkpoint).}",
          r"\label{fig:curves}", r"\end{figure}", ""]
    return L


def fig_curves(res: dict, runs_dir: str, path: str):
    plt = _style()
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.3), sharex=True, sharey=True)
    shades = ["#86b6ef", "#2a78d6", "#104281"]
    for ax, (v, k) in zip(axes.ravel(), VARIANT_KEYS.items()):
        for j, seed in enumerate((0, 1, 2)):
            p = os.path.join(runs_dir, f"{v}_s{seed}", "train_log.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            ep = [r["epoch"] for r in d["log"]]; r2 = [r["val_regional_r2"] for r in d["log"]]
            ax.plot(ep, r2, color=shades[j], lw=0.9, label=f"seed {seed}")
            be = d["best_epoch"]
            ax.plot([be], [r2[be - 1]], marker="o", ms=3.5, color=shades[j], mec="white", mew=0.6, zorder=3)
        ax.set_title(k.replace("ADmod without ", "w/o ").replace("ADmod with ", "").replace("Plain 3D U-Net", "plain 3D U-Net"), fontsize=7.5, loc="left")
        ax.set_ylim(-0.1, 0.55)
    axes.ravel()[-1].axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(h, l, loc="center", fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("epoch", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("val. $R^2$", fontsize=7)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def supp_repro() -> list[str]:
    import platform
    import numpy, scipy, sklearn, torch
    tt = lambda x: r"\texttt{" + x.replace("_", r"\_") + "}"
    full = lambda f: open(os.path.join("protocol", f)).read().split()[0]
    dig = [l.split() for l in open("protocol/CONFIRM_COHORT_DIGESTS.txt").read().strip().splitlines()]
    L = [r"\textbf{Software.} Python " + platform.python_version() + f", PyTorch {torch.__version__.split('+')[0]}, NumPy {numpy.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}.",
         "", r"\textbf{Hardware.} Two cores of an Intel Xeon processor with AMX units; no GPU. Training used bfloat16 autocast; every evaluation used float32.",
         "", r"\textbf{Protocol commits} in the restored repository (included with the code; the original repository was lost, see below):", r"\begin{itemize}",
         r"\item protocol and training code: " + tt(full("FREEZE_COMMIT.txt")),
         r"\item confirmatory analysis code and post hoc analysis code: " + tt(full("ANALYSIS_COMMIT.txt")),
         r"\item content digests of the confirmatory cohorts: " + tt(full("COHORT_COMMIT.txt")), r"\end{itemize}",
         r"\textbf{Confirmatory cohort content digests} (SHA-256 over every array's name, type, shape and bytes; \path{admod.dataset.array_digest}):", r"\begin{itemize}"]
    for h, f in dig:
        L.append(r"\item " + tt(os.path.basename(f)) + r": {\footnotesize" + tt(h) + "}")
    L += [r"\end{itemize}", "", r"\textbf{Loss and restoration of the original repository.} " + open("protocol/RESTORATION.tex").read().strip(),
          "", r"\textbf{Commands.} \texttt{python -m admod.dataset} (training and exploratory cohorts); \texttt{./run\_ax.sh} (sensitivity cohorts and all 33 training runs, resumable); \texttt{python -m admod.dataset --confirm}; \texttt{python -m admod.posthoc} (checks); \texttt{python -m admod.confirm}; \texttt{python -m admod.confirm --sensitivity}; \texttt{python -m admod.posthoc --analyze}; \texttt{python -m admod.make\_tex\_ax}. Seeds: generator seeds in Table~II of the main text; model seeds 0, 1, 2; bootstrap seed 2027.",
          "", r"\textbf{Released files.} Code, protocol, git history, trained checkpoints (\path{best.pt} for all 33 models), training logs, \path{results/confirm.json}, \path{results/sensitivity.json}, \path{results/posthoc.json}, and per-subject regional predictions of every method and seed on C1 to C3 (\path{results/confirm_regional.npz})."]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="../paper_access")
    ap.add_argument("--runs", default="runs_ax")
    a = ap.parse_args()
    res = json.load(open(os.path.join(a.results, "confirm.json")))
    opt = lambda f: json.load(open(os.path.join(a.results, f))) if os.path.exists(os.path.join(a.results, f)) else None
    sens, post = opt("sensitivity.json"), opt("posthoc.json")
    cal = json.load(open(os.path.join(a.results, "calibration.json")))
    pf = {prec: os.path.join("runs_parity", prec, "ax_full_s1", "train_log.json") for prec in ("bf16", "fp32")}
    parity = {prec: json.load(open(f)) for prec, f in pf.items()} if all(map(os.path.exists, pf.values())) else None
    commits = {name: open(os.path.join("protocol", f)).read().split()[0][:7]
               for name, f in (("commitProtocol", "FREEZE_COMMIT.txt"), ("commitAnalysis", "ANALYSIS_COMMIT.txt"),
                               ("commitCohorts", "COHORT_COMMIT.txt"))}
    os.makedirs(os.path.join(a.out, "supp"), exist_ok=True)
    w = lambda name, lines: open(os.path.join(a.out, name), "w").write("\n".join(lines) + "\n")
    w("numbers_ax.tex", ["% generated by admod/make_tex_ax.py; do not edit"] + number_macros(res, sens, post, cal, parity, commits))
    w("table_primary.tex", table_primary(res)); w("table_methods.tex", table_methods(res))
    w("table_ablation.tex", table_ablation(res)); w("table_variants.tex", table_variants(res))
    fig_forest(res, os.path.join(a.out, "fig_forest.pdf")); fig_gap(res, os.path.join(a.out, "fig_gap.pdf"))
    if sens:
        w("table_sens.tex", table_sens(sens)); fig_sens(sens, os.path.join(a.out, "fig_sens.pdf"))
    w("supp/supp_tables.tex", supp_tables(res, a.runs, post, os.path.join(a.out, "supp")))
    w("supp/supp_repro.tex", supp_repro())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
