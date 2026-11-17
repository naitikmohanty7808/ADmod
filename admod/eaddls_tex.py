"""Macros and tables for Supplementary Section S1 from results/eaddls.json.

    python -m admod.eaddls_tex --out ../paper_access/supp
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

MODELS = ["Logistic regression", "SVM (RBF)", "Decision tree", "Random forest", "AdaBoost", "EADDLS (MLP)"]
MS = {"Logistic regression": "LR", "SVM (RBF)": "SVM", "Decision tree": "Tree", "Random forest": "RF",
      "AdaBoost": "Ada", "EADDLS (MLP)": "MLP"}
FS = {"demo_mri": "DemoMri", "demo": "Demo", "mri": "Mri"}
TASK = {"detection": "Det", "conversion": "Conv"}


def f2(x):
    return f"{x:.2f}"


def p0(x):
    return f"{100 * x:.0f}\\%"


def macros(r: dict) -> list[str]:
    L = []
    m = lambda k, v: L.append(rf"\newcommand{{\ea{k}}}{{{v}}}")
    c = r["cohort"]
    m("NSubj", c["n_subjects"]); m("NVisits", c["n_visits"])
    m("NNd", c["groups"][0]); m("NDem", c["groups"][1]); m("NConv", c["groups"][2])
    rp = r["replica"]
    m("VonePos", rp["n_pos"]); m("VoneNeg", rp["n_neg"]); m("RuleAcc", p0(rp["rule_cdr_acc"]))
    for fs, K in (("v1_cdr", "Cdr"), ("v1_nocdr", "NoCdr")):
        v = list(rp[fs].values())
        m(f"Vone{K}AccMin", p0(min(v))); m(f"Vone{K}AccMax", p0(max(v)))
        m(f"Vone{K}AccLR", p0(rp[fs]["Logistic regression"])); m(f"Vone{K}AccMLP", p0(rp[fs]["EADDLS (MLP)"]))
    lk = list(r["leak_detection_cdr_auc"].values())
    m("LeakMin", f"{min(lk):.3f}")
    for t, T in TASK.items():
        d = r["auc"][t]
        m(f"{T}Pos", d["n_pos"]); m(f"{T}Neg", d["n_neg"])
        for fs, F in FS.items():
            for nm, v in d[fs].items():
                m(f"Auc{T}{F}{MS[nm]}", f2(v["auc"])); m(f"Lo{T}{F}{MS[nm]}", f2(v["ci"][0])); m(f"Hi{T}{F}{MS[nm]}", f2(v["ci"][1]))
        best = max(d["demo_mri"].items(), key=lambda kv: kv[1]["auc"])
        m(f"Best{T}", best[0]); m(f"BestAuc{T}", f2(best[1]["auc"]))
    vs = r["visit_split"]
    m("VisitRows", vs["n_rows"])
    gains = {nm: vs[nm]["visit_folds"] - vs[nm]["subject_folds"] for nm in MODELS}
    m("VisitGainMin", f2(min(gains.values()))); m("VisitGainMax", f2(max(gains.values())))
    top = max(gains, key=gains.get)
    m("VisitTop", top); m("VisitTopV", f2(vs[top]["visit_folds"])); m("VisitTopS", f2(vs[top]["subject_folds"]))
    gt = r["ground_truth"]
    m("GtN", gt["n_cohorts"]); m("PopN", f"{gt['pop_n']:,}".replace(",", "{,}"))
    for t, T in TASK.items():
        cv, tr = np.array(gt["tasks"][t]["cv"]), np.array(gt["tasks"][t]["true"])
        m(f"GtTrue{T}", f2(tr.mean())); m(f"GtTrueSD{T}", f2(tr.std(ddof=1)))
        m(f"GtCvMin{T}", f2(cv.min())); m(f"GtCvMax{T}", f2(cv.max()))
        m(f"GtCvLo{T}", f2(np.percentile(cv, 5))); m(f"GtCvHi{T}", f2(np.percentile(cv, 95)))
        m(f"GtGeSeven{T}", int((cv >= 0.70).sum())); m(f"GtCorr{T}", f2(np.corrcoef(cv, tr)[0, 1]))
        m(f"GtOver{T}", f2(float(np.mean(cv - tr))))
    ls = r["large_sample"]
    m("BigN", f"{ls['n_train']:,}".replace(",", "{,}"))
    wins = sum(v["EADDLS (MLP)"] > v["Logistic regression"] for v in ls["tasks"].values())
    m("MlpRefWins", ["none", "one", "two", "three", "all"][wins])
    for k, v in ls["tasks"].items():
        K = "".join(w.capitalize() for w in k.replace("demo_mri", "demomri").split("_"))
        m(f"Ref{K}LR", f2(v["Logistic regression"])); m(f"Ref{K}MLP", f2(v["EADDLS (MLP)"]))
    return L


def table_results(r: dict) -> list[str]:
    rp, d, vs = r["replica"], r["auc"], r["visit_split"]
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{S1 results on the synthetic OASIS-2-like study cohort. Replica: accuracy with the earlier version's labels "
         r"(demented and converters vs.\ nondemented) and inputs, mean over 20 repeats of 5-fold cross-validation. "
         r"Detection and conversion: AUC of out-of-fold scores (demographics + MRI; no CDR, no MMSE) with 95\% subject-bootstrap interval. "
         r"Split leakage: detection AUC with one row per visit, visit-level vs.\ subject-level folds.}",
         r"\label{tab:eaddls}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Replica accuracy} & Detection & Conversion & \multicolumn{2}{c}{Split leakage (AUC)}\\",
         r"Model & with CDR & without CDR & AUC & AUC & visit folds & subject folds\\", r"\midrule"]
    for nm in MODELS:
        a = d["detection"]["demo_mri"][nm]; b = d["conversion"]["demo_mri"][nm]
        T.append(f"{nm} & {p0(rp['v1_cdr'][nm])} & {p0(rp['v1_nocdr'][nm])} & "
                 f"{f2(a['auc'])} ({f2(a['ci'][0])}--{f2(a['ci'][1])}) & {f2(b['auc'])} ({f2(b['ci'][0])}--{f2(b['ci'][1])}) & "
                 f"{f2(vs[nm]['visit_folds'])} & {f2(vs[nm]['subject_folds'])}\\\\")
    T += [r"\midrule", f"Rule CDR $>$ 0 & {p0(rp['rule_cdr_acc'])} & & & & & \\\\", r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return T


def table_cohort(r: dict) -> list[str]:
    s = r["strata"]
    g = [("nondemented", r["cohort"]["groups"][0]), ("demented", r["cohort"]["groups"][1]), ("converter", r["cohort"]["groups"][2])]
    row = lambda f, nd=1: " & ".join(f"{s[k][f][0]:.{nd}f} ({s[k][f][1]:.{nd}f})" for k, _ in g)
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{First-visit values of the simulated S1 study cohort, mean (SD). Group sizes follow the published OASIS-2 design; "
         r"all distributions are assumed (Table~\ref{tab:eaassumed}). Demographics and eTIV do not depend on group.}",
         r"\label{tab:cohort}", r"\begin{tabular}{lccc}", r"\toprule",
         "& " + " & ".join(f"{k.capitalize()} ($n={n}$)" for k, n in g) + r"\\", r"\midrule",
         f"Age (y) & {row('age')}\\\\", f"Education (y) & {row('educ')}\\\\", f"SES (1--5) & {row('ses')}\\\\",
         "Female & " + " & ".join(p0(s[k]['female']) for k, _ in g) + r"\\",
         f"MMSE & {row('mmse')}\\\\", f"nWBV & {row('nwbv', 3)}\\\\", f"eTIV (cc) & {row('etiv', 0)}\\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return T


def table_assumed(r: dict) -> list[str]:
    A = r["assumed"]
    rows = [
        ("Female fraction", f"{A['female_fraction']:.2f}"),
        ("Age at first visit (y)", "normal, mean %.1f, SD %.1f, clipped to 60--92" % tuple(A["age_first_visit_mean_sd"])),
        ("Education (y)", "normal, mean %.1f, SD %.1f, rounded, 6--23" % tuple(A["education_years_mean_sd"])),
        ("SES", r"$3.4-(\mathrm{educ}-14.5)/2.5+\mathcal{N}(0,0.8^2)$, rounded, 1--5"),
        ("Visits per subject (2, 3, 4, 5)", ", ".join(f"{p:.2f}" for p in A["visits_2_3_4_5_prob"])),
        ("Visit interval (y)", "lognormal, median %.1f, $\\sigma$ %.2f, clipped to 0.5--4.7" % tuple(A["visit_interval_lognormal_median_sigma"])),
        ("Demented: years since onset at first visit", "uniform %.1f--%.1f" % tuple(A["demented_years_since_onset_at_first_visit"])),
        ("Converters: onset", "uniform within 5--95\\% of follow-up"),
        ("CDR 0.5 / 1 / 2 from onset (y)", " / ".join(f"{x:g}" for x in A["cdr_steps_years_since_onset"])),
        ("eTIV, male / female (cc)", "normal (%.0f, %.0f) / (%.0f, %.0f)" % (*A["etiv_male_female_mean_sd"][0], *A["etiv_male_female_mean_sd"][1])),
        ("nWBV at age 75", ("normal, mean %.3f, SD %.3f; age slope %.4f\\,y$^{-1}$" % (*A["nwbv_at_75_mean_sd"], A["nwbv_age_slope_per_year"])).replace("-", "$-$").replace("$^{$-$1}$", "$^{-1}$")),
        ("nWBV aging rate (y$^{-1}$)", ("normal, mean %.4f, SD %.4f" % tuple(A["nwbv_aging_rate_mean_sd"])).replace("-", "$-$")),
        ("nWBV disease loss $A_i/(1+e^{-(s-m)/w})$", "$A_i$ gamma, shape %.0f, mean %.3f; $m=%g$, $w=%g$\\,y" % (*A["nwbv_disease_amplitude_gamma_shape_mean"], *A["nwbv_disease_midpoint_width"])),
        ("nWBV measurement noise", f"{100 * A['nwbv_relative_noise']:.1f}\\% relative"),
        ("MMSE", "%.1f minus decline from %.1f\\,y before onset at a gamma rate (shape %.0f, mean %.1f\\,y$^{-1}$)" % (A["mmse_ceiling"], A["mmse_decline_start_before_onset"], *A["mmse_decline_gamma_shape_mean_per_year"])),
        ("MMSE test error SD", "%.1f (CDR 0), %.1f (CDR $>$ 0); rounded, 0--30" % tuple(A["mmse_error_sd_unimpaired_impaired"])),
    ]
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Generative assumptions of the S1 cohort. None is fitted to data; only the group sizes and the minimum age of 60 years come from the OASIS-2 publication~\cite{marcus2010}.}",
         r"\label{tab:eaassumed}", r"\begin{tabular}{p{0.36\textwidth}p{0.58\textwidth}}", r"\toprule", r"Quantity & Assumption\\", r"\midrule"]
    T += [f"{a} & {b}\\\\" for a, b in rows]
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/eaddls.json")
    ap.add_argument("--out", default="../paper_access/supp")
    a = ap.parse_args()
    r = json.load(open(a.results))
    w = lambda n, L: open(os.path.join(a.out, n), "w").write("\n".join(L) + "\n")
    w("numbers_ea.tex", ["% generated by admod/eaddls_tex.py; do not edit"] + macros(r))
    w("table_eaddls.tex", table_results(r)); w("table_cohort.tex", table_cohort(r)); w("table_ea_assumed.tex", table_assumed(r))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
