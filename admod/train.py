"""Train ADmod variants on SYNTHETIC training subjects. Keeps the checkpoint with the best
validation loss. The regional prior is fit on training subjects only and saved with the model.

    python -m admod.train --variant admod2 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .baselines import RegionalLinear2, RegionalTrend, from_state
from .metrics import summarize
from .models import ADmod, build_features, count_params, loss_weights

VARIANTS = {
    # main model: trained 60 epochs (python -m admod.train --variant admod2_noprior_e60 --epochs 60)
    "admod2_noprior_e60": {"n_scans": 2, "prior": None, "label": "ADmod (2 scans)"},
    # 2 x 2 ablations: trained 30 epochs
    "admod2": {"n_scans": 2, "prior": "RegionalLinear2", "label": "ADmod + regional prior (2 scans) [30 ep]"},
    "admod1": {"n_scans": 1, "prior": "RegionalTrend", "label": "ADmod + regional prior (1 scan) [30 ep]"},
    "admod2_noprior": {"n_scans": 2, "prior": None, "label": "ADmod (2 scans) [30 ep]"},
    "admod1_noprior": {"n_scans": 1, "prior": None, "label": "ADmod (1 scan) [30 ep]"},
}
# IEEE Access revision: one training policy for every variant (100 epochs, cosine to zero, best
# validation-loss checkpoint, bfloat16 autocast). Each ablation removes one component of ax_full.
_AX = {"prior_ch": False}
AX_VARIANTS = {
    "ax_full": ({}, 2, "ADmod"),
    "ax_plain": ({"global_ctx": False, "region_maps": False, "coords": False, "head": "direct"}, 2, "Plain 3D U-Net"),
    "ax_no_gap": ({"gap_input": False}, 2, "ADmod without gap input"),
    "ax_no_global": ({"global_ctx": False}, 2, "ADmod without global context"),
    "ax_no_region": ({"region_maps": False}, 2, "ADmod without region maps"),
    "ax_one_scan": ({}, 1, "ADmod with one scan"),
    "ax_direct": ({"head": "direct"}, 2, "ADmod with direct head"),
}
for _k, (_a, _n, _lab) in AX_VARIANTS.items():
    VARIANTS[_k] = {"n_scans": _n, "prior": None, "label": _lab, "arch": dict(_AX, **_a)}
PRIORS = {"RegionalLinear2": RegionalLinear2, "RegionalTrend": RegionalTrend}


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prior_maps(prior_model, cohort: dict) -> np.ndarray:
    """(B,2,N,N,N): [gap-independent fix, yearly rate] from the regional prior, zeros if none."""
    if prior_model is None:
        return np.zeros((len(cohort["dt"]), 2) + cohort["baseline"].shape[1:], np.float32)
    fix, rate = prior_model.maps(cohort)
    return np.ascontiguousarray(np.stack([fix, rate], 1).astype(np.float32))


@torch.no_grad()
def predict(model: ADmod, cohort: dict, prior_model, batch: int = 8) -> np.ndarray:
    model.eval()
    pri = prior_maps(prior_model, cohort)
    out = []
    for i in range(0, len(cohort["dt"]), batch):
        sl = slice(i, i + batch)
        f = torch.from_numpy(build_features(cohort, model.n_scans, sl))
        pred, _ = model(f, torch.from_numpy(cohort["dt"][sl].astype(np.float32)), torch.from_numpy(pri[sl]))
        out.append(pred[:, 0].numpy())
    return np.concatenate(out)


def load_model(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ADmod(n_scans=ck["n_scans"], width=ck["width"], arch=ck.get("arch"))
    model.load_state_dict(ck["state"])
    model.eval()
    prior = from_state(ck["prior"]) if ck["prior"] else None
    return model, prior, ck


def train(variant: str, seed: int, data: str, out: str, epochs: int, lr: float, width: int, batch: int,
          bf16: bool = False) -> dict:
    cfg = VARIANTS[variant]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)

    tr, va = load(os.path.join(data, "train.npz")), load(os.path.join(data, "val.npz"))
    assert bool(tr["synthetic"]) and bool(va["synthetic"])
    assert not set(tr["ids"]) & set(va["ids"]), "train/val subject overlap"

    pm = PRIORS[cfg["prior"]]().fit(tr) if cfg["prior"] else None
    n_scans = cfg["n_scans"]
    Xtr = torch.from_numpy(build_features(tr, n_scans))
    Ptr = torch.from_numpy(prior_maps(pm, tr))
    Ytr = torch.from_numpy(tr["followup"].astype(np.float32))[:, None]
    Wtr = torch.from_numpy(loss_weights(tr["labels"]))
    Dtr = torch.from_numpy(tr["dt"].astype(np.float32))
    w_va = loss_weights(va["labels"])[:, 0]

    arch = cfg.get("arch")
    model = ADmod(n_scans=n_scans, width=width, arch=arch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"[{variant} seed={seed}] params={count_params(model):,} train={len(Dtr)} val={len(va['dt'])}", flush=True)

    best, log, n = float("inf"), [], len(Dtr)
    t_start, start_ep, prev_sec = time.time(), 1, 0.0
    last = os.path.join(out, "last.pt")
    if os.path.exists(last):  # resume an interrupted run exactly where it stopped
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["np_rng"]; torch.set_rng_state(st["torch_rng"])
        best, log, start_ep, prev_sec = st["best"], st["log"], st["epoch"] + 1, st["sec"]
        print(f"resumed after epoch {st['epoch']}", flush=True)
    for ep in range(start_ep, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(n))
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            f, p, y, w, d = Xtr[idx], Ptr[idx], Ytr[idx], Wtr[idx], Dtr[idx]
            if rng.random() < 0.5:  # left/right mirror; template coordinates are not mirrored
                f, p, y, w = (t.flip(2) for t in (f, p, y, w))
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                pred, _ = model(f, d, p)
            loss = (w * (pred.float() - y) ** 2).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        pv = predict(model, va, pm)
        vloss = float((w_va * (pv - va["followup"]) ** 2).sum() / w_va.sum())
        s = summarize(pv, va["followup"], va["baseline"], va["labels"])
        rec = {"epoch": ep, "train_loss": round(tot / n, 6), "val_loss": round(vloss, 6),
               "val_regional_r2": round(s["regional_r2"], 4), "sec": round(time.time() - t0, 1)}
        log.append(rec)
        flag = ""
        if vloss < best:
            best = vloss
            torch.save({"state": model.state_dict(), "n_scans": n_scans, "width": width, "epoch": ep,
                        "variant": variant, "seed": seed, "prior": pm.state() if pm else None,
                        "lr": lr, "epochs": epochs, "batch": batch, "arch": arch, "bf16": bf16},
                       os.path.join(out, "best.pt"))
            flag = "  *saved*"
        print(json.dumps(rec) + flag, flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "best": best,
                    "log": log, "epoch": ep, "sec": prev_sec + time.time() - t0}, last + ".tmp")
        os.replace(last + ".tmp", last)
        prev_sec += time.time() - t0

    summary = {"variant": variant, "seed": seed, "best_val_loss": best,
               "best_epoch": min(log, key=lambda r: r["val_loss"])["epoch"], "epochs": epochs,
               "bf16": bf16, "params": count_params(model), "n_train": int(n), "n_val": int(len(va["dt"])),
               "minutes": round(sum(r["sec"] for r in log) / 60, 1), "log": log}
    with open(os.path.join(out, "train_log.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if os.path.exists(last):
        os.remove(last)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="admod2", choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast for training (AMX CPUs); eval stays fp32")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for s in args.seeds:
        if os.path.exists(os.path.join(args.runs, f"{args.variant}_s{s}", "train_log.json")):
            print(f"{args.variant} seed {s}: already finished, skipping", flush=True)
            continue
        train(args.variant, s, args.data, os.path.join(args.runs, f"{args.variant}_s{s}"),
              args.epochs, args.lr, args.width, args.batch, args.bf16)


if __name__ == "__main__":
    main()
