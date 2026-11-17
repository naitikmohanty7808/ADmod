"""Reanalysis of the earlier version's classification study (EADDLS) on a synthetic OASIS-2-like cohort.

All data are simulated. No OASIS data are read or copied. Only the published design counts of OASIS-2
(150 subjects aged 60 to 96, 72 nondemented throughout, 64 demented at the first visit, 14 converters,
373 visits; Marcus et al. 2010) are used. Every distribution below is an assumption and is listed in
ASSUMED so that the supplement can print it.

Run: python -m admod.eaddls  (writes results/eaddls.json)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DESIGN = {"n_nondemented": 72, "n_demented": 64, "n_converter": 14, "age_range": (60, 96), "n_visits": 373}

# Assumed generative parameters (not fitted to any data). Units: years, nWBV fraction, MMSE points, cc.
ASSUMED = {
    "female_fraction": 0.6,
    "age_first_visit_mean_sd": (76.0, 7.5),
    "education_years_mean_sd": (14.5, 2.9),
    "visits_2_3_4_5_prob": (0.62, 0.27, 0.08, 0.03),
    "visit_interval_lognormal_median_sigma": (1.8, 0.35),
    "demented_years_since_onset_at_first_visit": (0.2, 6.0),
    "cdr_steps_years_since_onset": (0.0, 3.0, 6.0),
    "etiv_male_female_mean_sd": ((1600.0, 150.0), (1420.0, 130.0)),
    "nwbv_at_75_mean_sd": (0.735, 0.030),
    "nwbv_age_slope_per_year": -0.0030,
    "nwbv_aging_rate_mean_sd": (-0.0035, 0.0020),
    "nwbv_disease_amplitude_gamma_shape_mean": (2.0, 0.040),
    "nwbv_disease_midpoint_width": (0.0, 3.0),
    "nwbv_relative_noise": 0.003,
    "mmse_ceiling": 29.2,
    "mmse_decline_gamma_shape_mean_per_year": (2.0, 2.2),
    "mmse_decline_start_before_onset": 1.5,
    "mmse_error_sd_unimpaired_impaired": (0.9, 1.8),
}

FEATURES = {
    "demo": ["age", "male", "educ", "ses"],
    "mri": ["etiv", "nwbv"],
    "demo_mri": ["age", "male", "educ", "ses", "etiv", "nwbv"],
    "v1_cdr": ["educ", "ses", "mmse", "cdr", "etiv", "nwbv", "asf"],
    "v1_nocdr": ["educ", "ses", "mmse", "etiv", "nwbv", "asf"],
    "v1_nocdr_nommse": ["educ", "ses", "etiv", "nwbv", "asf"],
}


def _cdr(s: np.ndarray) -> np.ndarray:
    a, b, c = ASSUMED["cdr_steps_years_since_onset"]
    return np.select([s >= c, s >= b, s >= a], [2.0, 1.0, 0.5], 0.0)


def make_cohort(rng: np.random.Generator, n_nd: int = 72, n_dem: int = 64, n_conv: int = 14) -> dict:
    """One row per visit. group: 0 nondemented, 1 demented at first visit, 2 converter."""
    A = ASSUMED
    group = np.r_[np.zeros(n_nd), np.ones(n_dem), np.full(n_conv, 2)].astype(int)
    n = len(group)
    male = (rng.random(n) > A["female_fraction"]).astype(float)
    lo, hi = DESIGN["age_range"]
    age0 = np.clip(rng.normal(*A["age_first_visit_mean_sd"], n), lo, hi - 4)
    educ = np.clip(np.round(rng.normal(*A["education_years_mean_sd"], n)), 6, 23)
    ses = np.clip(np.round(3.4 - (educ - 14.5) / 2.5 + rng.normal(0, 0.8, n)), 1, 5)
    (mm, ms), (fm, fs) = A["etiv_male_female_mean_sd"]
    etiv = np.where(male > 0, rng.normal(mm, ms, n), rng.normal(fm, fs, n))
    nv = rng.choice([2, 3, 4, 5], size=n, p=A["visits_2_3_4_5_prob"])
    med, sig = A["visit_interval_lognormal_median_sigma"]
    times = [np.r_[0.0, np.cumsum(np.clip(med * np.exp(sig * rng.standard_normal(k - 1)), 0.5, 4.7))] for k in nv]
    s0 = np.full(n, -np.inf)                       # years since clinical onset at the first visit
    dem = group == 1
    s0[dem] = rng.uniform(*A["demented_years_since_onset_at_first_visit"], dem.sum())
    for i in np.where(group == 2)[0]:              # onset strictly inside the follow-up
        s0[i] = -rng.uniform(0.05, 0.95) * times[i][-1]
    nw75 = rng.normal(*A["nwbv_at_75_mean_sd"], n)
    aging = rng.normal(*A["nwbv_aging_rate_mean_sd"], n)
    k, mean = A["nwbv_disease_amplitude_gamma_shape_mean"]
    amp = np.where(group > 0, rng.gamma(k, mean / k, n), 0.0)
    m_mid, w = A["nwbv_disease_midpoint_width"]
    k2, mean2 = A["mmse_decline_gamma_shape_mean_per_year"]
    mrate = np.where(group > 0, rng.gamma(k2, mean2 / k2, n), 0.0)
    rows = {c: [] for c in ("subject", "group", "visit", "years", "age", "male", "educ", "ses", "etiv", "asf",
                            "nwbv", "mmse", "cdr")}
    for i in range(n):
        t = times[i]
        s = s0[i] + t
        dis = amp[i] / (1 + np.exp(-(s - m_mid) / w)) if group[i] > 0 else np.zeros_like(t)
        nwbv = nw75[i] + A["nwbv_age_slope_per_year"] * (age0[i] - 75) + aging[i] * t - dis
        nwbv = nwbv * (1 + A["nwbv_relative_noise"] * rng.standard_normal(len(t)))
        start = A["mmse_decline_start_before_onset"]
        decline = mrate[i] * np.clip(s + start, 0, None) if group[i] > 0 else np.zeros_like(t)
        cdr = _cdr(s) if group[i] > 0 else np.zeros_like(t)
        lo_e, hi_e = A["mmse_error_sd_unimpaired_impaired"]
        sd = np.where(cdr > 0, hi_e, lo_e)
        mmse = np.clip(np.round(A["mmse_ceiling"] - decline + sd * rng.standard_normal(len(t))), 0, 30)
        for j in range(len(t)):
            for c, v in (("subject", i), ("group", group[i]), ("visit", j), ("years", t[j]), ("age", age0[i] + t[j]),
                         ("male", male[i]), ("educ", educ[i]), ("ses", ses[i]), ("etiv", etiv[i]),
                         ("asf", 1755.0 / etiv[i]), ("nwbv", nwbv[j]), ("mmse", mmse[j]), ("cdr", cdr[j])):
                rows[c].append(v)
    out = {c: np.asarray(v, float) for c, v in rows.items()}
    out["subject"] = out["subject"].astype(int); out["group"] = out["group"].astype(int)
    return out


def first_visit(c: dict) -> dict:
    m = c["visit"] == 0
    return {k: v[m] for k, v in c.items()}


def check_design(c: dict) -> dict:
    """The CDR sequence must reproduce the group definitions, as in OASIS-2."""
    g = first_visit(c)["group"]
    cd0 = first_visit(c)["cdr"]
    later = np.array([c["cdr"][c["subject"] == i].max() for i in range(len(g))])
    assert np.all((cd0 > 0) == (g == 1)), "demented group must have CDR > 0 at the first visit"
    assert np.all((later > 0) == (g > 0)), "converters must reach CDR > 0 later, nondemented never"
    return {"n_subjects": int(len(g)), "n_visits": int(len(c["subject"])),
            "groups": [int((g == k).sum()) for k in range(3)]}


def models(seed: int) -> dict:
    sc = lambda m: make_pipeline(StandardScaler(), m)
    return {
        "Logistic regression": sc(LogisticRegression(C=1.0, max_iter=2000)),
        "SVM (RBF)": sc(SVC(C=1.0, kernel="rbf", probability=False)),
        "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=seed),
        "EADDLS (MLP)": sc(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01, max_iter=4000, random_state=seed)),
    }


def _score(m, X):
    if hasattr(m, "decision_function"):
        return m.decision_function(X)
    return m.predict_proba(X)[:, 1]


def cv_scores(X, y, name, repeats=20, groups=None, seed=0, subject_folds=True):
    """Out-of-fold scores averaged (as ranks) over repeats, and mean out-of-fold accuracy."""
    n = len(y)
    ranks = np.zeros(n); acc = []
    for r in range(repeats):
        oof = np.zeros(n); pred = np.zeros(n)
        if groups is not None and subject_folds:
            u = np.unique(groups); perm = np.random.default_rng(seed + r).permutation(len(u))
            remap = dict(zip(u, perm)); g = np.array([remap[v] for v in groups])
            splits = GroupKFold(n_splits=5).split(X, y, g)
        else:
            splits = StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y)
        for tr, te in splits:
            m = models(seed + r)[name].fit(X[tr], y[tr])
            oof[te] = _score(m, X[te]); pred[te] = m.predict(X[te])
        ranks += np.argsort(np.argsort(oof)); acc.append(float((pred == y).mean()))
    return ranks / repeats, float(np.mean(acc))


def boot_auc(y, s, rng, B=2000, groups=None):
    est = roc_auc_score(y, s)
    if groups is None:
        groups = np.arange(len(y))
    u = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in u}
    vals = []
    for _ in range(B):
        take = np.concatenate([idx[g] for g in rng.choice(u, len(u))])
        if len(np.unique(y[take])) == 2:
            vals.append(roc_auc_score(y[take], s[take]))
    return float(est), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def X_of(c, feats):
    return np.column_stack([c[f] for f in feats])


def task_labels(c1: dict, task: str):
    g = c1["group"]
    if task == "detection":
        return np.ones(len(g), bool), (g == 1).astype(int)
    if task == "conversion":
        keep = g != 1
        return keep, (g[keep] == 2).astype(int)
    if task == "v1":                               # earlier version: demented and converters vs nondemented
        return np.ones(len(g), bool), (g > 0).astype(int)
    raise ValueError(task)


def run(seed: int = 4100, repeats: int = 20, n_gt: int = 200, pop_n: int = 20000, big_n: int = 30000) -> dict:
    rng = np.random.default_rng(seed)
    study = make_cohort(rng)
    res = {"design": DESIGN, "assumed": ASSUMED, "seed": seed, "repeats": repeats, "cohort": check_design(study)}
    c1 = first_visit(study)
    g = c1["group"]
    res["strata"] = {k: {f: [float(c1[f][g == gi].mean()), float(c1[f][g == gi].std(ddof=1))]
                         for f in ("age", "educ", "ses", "mmse", "nwbv", "etiv")} | {"female": float(1 - c1["male"][g == gi].mean())}
                     for k, gi in (("nondemented", 0), ("demented", 1), ("converter", 2))}
    brng = np.random.default_rng(seed + 1)
    names = list(models(0))
    # Replica of the earlier version: its labels and inputs, accuracy
    keep, y = task_labels(c1, "v1")
    res["replica"] = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                      "rule_cdr_acc": float(((c1["cdr"] > 0).astype(int) == y).mean())}
    for fs in ("v1_cdr", "v1_nocdr", "v1_nocdr_nommse"):
        res["replica"][fs] = {nm: cv_scores(X_of(c1, FEATURES[fs]), y, nm, repeats, seed=seed)[1] for nm in names}
    # Leak condition: converters counted as not demented, CDR as an input
    _, yd = task_labels(c1, "detection")
    res["leak_detection_cdr_auc"] = {nm: roc_auc_score(yd, cv_scores(X_of(c1, FEATURES["v1_cdr"]), yd, nm, 2, seed=seed)[0])
                                     for nm in names}
    # Detection and conversion without CDR
    res["auc"] = {}
    for task in ("detection", "conversion"):
        keep, y = task_labels(c1, task)
        cc = {k: v[keep] for k, v in c1.items()}
        res["auc"][task] = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
        for fs in ("demo_mri", "demo", "mri"):
            todo = names if fs == "demo_mri" else ["Logistic regression", "EADDLS (MLP)"]
            res["auc"][task][fs] = {}
            for nm in todo:
                s, _ = cv_scores(X_of(cc, FEATURES[fs]), y, nm, repeats, seed=seed)
                est, ci = boot_auc(y, s, brng)
                res["auc"][task][fs][nm] = {"auc": est, "ci": ci}
    # Split leakage: one row per visit, label = subject's group, visit-level vs subject-level folds
    yv = (study["group"] == 1).astype(int)
    Xv = X_of(study, FEATURES["demo_mri"])
    res["visit_split"] = {"n_rows": int(len(yv))}
    for nm in names:
        sv, _ = cv_scores(Xv, yv, nm, repeats, groups=study["subject"], seed=seed, subject_folds=False)
        ss, _ = cv_scores(Xv, yv, nm, repeats, groups=study["subject"], seed=seed, subject_folds=True)
        res["visit_split"][nm] = {"visit_folds": float(roc_auc_score(yv, sv)), "subject_folds": float(roc_auc_score(yv, ss))}
    # Ground truth: many study-sized cohorts against a large population with the same design proportions
    scale = pop_n // 150
    pop = first_visit(make_cohort(np.random.default_rng(seed + 2), 72 * scale, 64 * scale, 14 * scale))
    gt = {}
    for task in ("detection", "conversion"):
        kp, yp = task_labels(pop, task)
        Xp = X_of({k: v[kp] for k, v in pop.items()}, FEATURES["demo_mri"])
        cv_est, true = [], []
        for r in range(n_gt):
            cr = first_visit(make_cohort(np.random.default_rng(seed + 1000 + r)))
            kr, yr = task_labels(cr, task)
            Xr = X_of({k: v[kr] for k, v in cr.items()}, FEATURES["demo_mri"])
            s, _ = cv_scores(Xr, yr, "Logistic regression", 2, seed=seed + r)
            cv_est.append(float(roc_auc_score(yr, s)))
            m = models(0)["Logistic regression"].fit(Xr, yr)
            true.append(float(roc_auc_score(yp, _score(m, Xp))))
        gt[task] = {"cv": cv_est, "true": true}
    res["ground_truth"] = {"n_cohorts": n_gt, "pop_n": int(len(pop["group"])), "tasks": gt}
    # Large-sample reference: is the MLP more accurate than logistic regression with ample data?
    scale_b = big_n // 150
    big = first_visit(make_cohort(np.random.default_rng(seed + 3), 72 * scale_b, 64 * scale_b, 14 * scale_b))
    ref = {}
    for task in ("detection", "conversion"):
        kb, yb = task_labels(big, task); kp, yp = task_labels(pop, task)
        for fs in ("demo_mri", "mri"):
            Xb = X_of({k: v[kb] for k, v in big.items()}, FEATURES[fs]); Xp = X_of({k: v[kp] for k, v in pop.items()}, FEATURES[fs])
            ref[f"{task}_{fs}"] = {nm: float(roc_auc_score(yp, _score(models(0)[nm].fit(Xb, yb), Xp)))
                                   for nm in ("Logistic regression", "EADDLS (MLP)")}
    res["large_sample"] = {"n_train": int(len(big["group"])), "tasks": ref}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/eaddls.json")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--n-gt", type=int, default=200)
    ap.add_argument("--add-nommse", action="store_true",
                    help="add the replica without CDR and MMSE to an existing results file (same cohort and seeds)")
    a = ap.parse_args()
    if a.add_nommse:
        res = json.load(open(a.out))
        c1 = first_visit(make_cohort(np.random.default_rng(res["seed"])))
        _, y = task_labels(c1, "v1")
        X = X_of(c1, FEATURES["v1_nocdr_nommse"])
        res["replica"]["v1_nocdr_nommse"] = {nm: cv_scores(X, y, nm, res["repeats"], seed=res["seed"])[1] for nm in models(0)}
        json.dump(res, open(a.out, "w"), indent=1)
        print(res["replica"]["v1_nocdr_nommse"])
        return
    res = run(repeats=a.repeats, n_gt=a.n_gt)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
