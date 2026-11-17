"""Generate synthetic cohorts. Each subject has three scans: prior, baseline, follow-up.

Forecasting task: given the prior and baseline scans (or the baseline alone), predict the follow-up.
All data produced here is SYNTHETIC and saved with `synthetic=True`.
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np

from .phantom import make_brain
from .amyloid import PATTERNS, shape_pattern, simulate, susceptibility_map, sample_params
from .pet import sample_tissue, scan

GRID = 32
PRIOR_GAP = (1.5, 3.0)   # years between prior and baseline scan
FORECAST_GAP = (1.0, 4.0)  # years between baseline and follow-up


def make_subject(args) -> dict:
    seed, pattern, gap_range, n, noise_scale = args[:5]
    max_shift = args[5] if len(args) > 5 else 0.3  # misregistration range, voxels
    opts = args[6] if len(args) > 6 else {}        # sensitivity settings: rho_sd, susc_gamma
    rng = np.random.default_rng(seed)
    brain = make_brain(rng, n)
    labels, gm = brain["labels"], brain["cortex_gm"]
    params = sample_params(rng, opts.get("rho_sd", 0.25))
    s = susceptibility_map(labels, gm, shape_pattern(PATTERNS[pattern], opts.get("susc_gamma", 1.0)), rng)
    t0 = round(float(rng.uniform(3.0, 28.0)), 1)
    g0 = round(float(rng.uniform(*PRIOR_GAP)), 1)
    g1 = round(float(rng.uniform(*gap_range)), 1)
    cp, c0, c1 = simulate(gm, s, params["rho"], params["eps"], params["D"], [t0 - g0, t0, t0 + g1])
    tissue = sample_tissue(rng)
    sp, sp_clean = scan(labels, cp, tissue, rng, noise_scale, max_shift_vox=max_shift)
    sb, sb_clean = scan(labels, c0, tissue, rng, noise_scale, max_shift_vox=max_shift)
    sf, sf_clean = scan(labels, c1, tissue, rng, noise_scale, max_shift_vox=max_shift)
    return {
        "prior": sp, "prior_clean": sp_clean,
        "baseline": sb, "baseline_clean": sb_clean,
        "followup": sf, "followup_clean": sf_clean,
        "labels": labels, "dt_prior": np.float32(g0), "dt": np.float32(g1), "t0": np.float32(t0),
        "rho": params["rho"], "eps": params["eps"], "D": params["D"], "resistant": params["resistant"],
    }


def generate_cohort(name: str, n_subjects: int, base_seed: int, pattern: str = "typical",
                    gap_range=FORECAST_GAP, n: int = GRID, noise_scale: float = 1.0, workers: int = 2,
                    max_shift: float = 0.3, opts: dict | None = None) -> dict:
    jobs = [(base_seed * 100_000 + i, pattern, gap_range, n, noise_scale, max_shift, opts or {})
            for i in range(n_subjects)]
    with Pool(workers) as pool:
        subs = pool.map(make_subject, jobs, chunksize=4)
    out = {k: np.stack([s[k] for s in subs]) for k in subs[0]}
    out["ids"] = np.array([f"{name}-{i:04d}" for i in range(n_subjects)])
    out["pattern"] = np.array(pattern)
    out["noise_scale"] = np.float32(noise_scale)
    out["max_shift"] = np.float32(max_shift)
    out["rho_sd"] = np.float32((opts or {}).get("rho_sd", 0.25))
    out["susc_gamma"] = np.float32((opts or {}).get("susc_gamma", 1.0))
    out["base_seed"] = np.int64(base_seed)
    out["synthetic"] = np.array(True)
    return out


def split_ids(ids: np.ndarray, fractions=(0.7, 0.1, 0.2), seed: int = 0) -> dict:
    """Split by subject. Each subject lands in exactly one split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_tr = int(round(fractions[0] * len(ids)))
    n_va = int(round(fractions[1] * len(ids)))
    return {"train": np.sort(perm[:n_tr]), "val": np.sort(perm[n_tr:n_tr + n_va]),
            "test": np.sort(perm[n_tr + n_va:])}


def subset(cohort: dict, idx: np.ndarray) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim > 0 else v) for k, v in cohort.items()}


def save(path: str, cohort: dict):
    np.savez_compressed(path, **cohort)


def main():
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC amyloid PET cohorts")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-main", type=int, default=520)
    ap.add_argument("--n-stress", type=int, default=100)
    ap.add_argument("--sweep", action="store_true", help="also make noise-sweep cohorts")
    ap.add_argument("--confirm", action="store_true", help="only make the confirmatory cohorts")
    ap.add_argument("--sensitivity", action="store_true", help="only make the sensitivity cohorts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.confirm or args.sensitivity:
        if args.sensitivity:
            sensitivity_cohorts(args.out)
        if args.confirm:
            confirm_cohorts(args.out)
        return

    main_c = generate_cohort("main", args.n_main, base_seed=1)
    for k, idx in split_ids(main_c["ids"]).items():
        save(os.path.join(args.out, f"{k}.npz"), subset(main_c, idx))
        print(f"{k}: {len(idx)} subjects", flush=True)

    save(os.path.join(args.out, "stress_atypical.npz"),
         generate_cohort("atypical", args.n_stress, base_seed=2, pattern="atypical"))
    print(f"stress_atypical: {args.n_stress} subjects (unseen spread pattern)", flush=True)
    save(os.path.join(args.out, "stress_long_gap.npz"),
         generate_cohort("long", args.n_stress, base_seed=3, gap_range=(6.0, 8.0)))
    print(f"stress_long_gap: {args.n_stress} subjects (6-8 year forecast; training used 1-4)", flush=True)

    if args.sweep:
        for m in (0.5, 1.0, 1.5, 2.0):
            save(os.path.join(args.out, f"sweep_train_{m}.npz"), generate_cohort("swtr", 240, 4, noise_scale=m))
            save(os.path.join(args.out, f"sweep_test_{m}.npz"), generate_cohort("swte", 120, 5, noise_scale=m))
            print(f"noise sweep {m}: 240 train / 120 test", flush=True)


def extra(out: str = "data"):
    """Cohorts added after the first review: a fresh long-horizon cohort (the original one was looked at
    while the regional baselines were designed) and a misregistration sweep (all other noise at x1)."""
    save(os.path.join(out, "stress_long_gap_fresh.npz"), generate_cohort("longfresh", 100, 7, gap_range=(6.0, 8.0)))
    print("stress_long_gap_fresh: 100 subjects", flush=True)
    for sh in (0.0, 0.1, 0.2, 0.3):
        save(os.path.join(out, f"misreg_train_{sh}.npz"), generate_cohort("mrtr", 240, 8, max_shift=sh))
        save(os.path.join(out, f"misreg_test_{sh}.npz"), generate_cohort("mrte", 120, 9, max_shift=sh))
        print(f"misregistration sweep {sh}: 240 train / 120 test", flush=True)


# ---------------------------------------------------------------- IEEE Access revision
# Confirmatory cohorts: new seeds, generated only after the protocol was frozen (protocol/protocol.json).
CONFIRM = {
    "confirm_id": {"n": 400, "seed": 21, "pattern": "typical", "gap": FORECAST_GAP},
    "confirm_atypical": {"n": 200, "seed": 22, "pattern": "atypical", "gap": FORECAST_GAP},
    "confirm_long": {"n": 200, "seed": 23, "pattern": "typical", "gap": (6.0, 8.0)},
}

# Sensitivity settings: ADmod is retrained (1 seed) and baselines refit for each. The noise and
# misregistration sweeps reuse the existing sweep cohorts (seeds 4/5 and 8/9); the four new
# settings use seeds 31-38. Training cohorts (240) are split 210 train / 30 validation by subject.
SENSITIVITY = {
    **{f"noise_{m}": {"src": (f"sweep_train_{m}.npz", f"sweep_test_{m}.npz")} for m in (0.5, 1.0, 1.5, 2.0)},
    **{f"misreg_{sh}": {"src": (f"misreg_train_{sh}.npz", f"misreg_test_{sh}.npz")} for sh in (0.0, 0.1, 0.2, 0.3)},
    "rate_sd_low": {"opts": {"rho_sd": 0.10}, "seeds": (31, 32)},
    "rate_sd_high": {"opts": {"rho_sd": 0.45}, "seeds": (33, 34)},
    "susc_flat": {"opts": {"susc_gamma": 0.5}, "seeds": (35, 36)},
    "susc_steep": {"opts": {"susc_gamma": 2.0}, "seeds": (37, 38)},
}


def confirm_cohorts(out: str = "data"):
    for name, c in CONFIRM.items():
        path = os.path.join(out, f"{name}.npz")
        if os.path.exists(path):
            print(f"{name}: exists, not regenerated", flush=True)
            continue
        save(path, generate_cohort(name, c["n"], c["seed"], pattern=c["pattern"], gap_range=c["gap"]))
        print(f"{name}: {c['n']} subjects (seed {c['seed']})", flush=True)


def sensitivity_cohorts(out: str = "data"):
    for name, c in SENSITIVITY.items():
        d = os.path.join(out, "sens", name)
        if os.path.exists(os.path.join(d, "test.npz")):
            continue
        os.makedirs(d, exist_ok=True)
        if "src" in c:
            with np.load(os.path.join(out, c["src"][0])) as z:
                tr = {k: z[k] for k in z.files}
            with np.load(os.path.join(out, c["src"][1])) as z:
                te = {k: z[k] for k in z.files}
        else:
            tr = generate_cohort(f"{name}-tr", 240, c["seeds"][0], opts=c["opts"])
            te = generate_cohort(f"{name}-te", 120, c["seeds"][1], opts=c["opts"])
        idx = split_ids(tr["ids"], fractions=(0.875, 0.125, 0.0), seed=0)
        save(os.path.join(d, "train.npz"), subset(tr, idx["train"]))
        save(os.path.join(d, "val.npz"), subset(tr, idx["val"]))
        save(os.path.join(d, "test.npz"), te)
        print(f"sens/{name}: {len(idx['train'])} train / {len(idx['val'])} val / {len(te['dt'])} test", flush=True)


def array_digest(path: str) -> str:
    """SHA-256 over every array's name, dtype, shape and bytes in sorted order. Unlike a hash of the
    .npz file, this does not depend on the zip timestamps, so a regenerated cohort can be checked."""
    import hashlib
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(z.files):
            a = np.ascontiguousarray(z[k])
            h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
