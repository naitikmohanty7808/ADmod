"""Amyloid build-up simulator (SYNTHETIC).

Model: a reaction-diffusion (Fisher-KPP) equation restricted to cortical gray matter.

    dc/dt = D * L(c) + rho * s(x) * c * (1 - c) + eps * s(x) * (1 - c)

    c      amyloid load per voxel, 0 (none) to 1 (saturated)
    L      spread to neighboring gray-matter voxels (graph Laplacian, no leak outside gray matter)
    s(x)   regional susceptibility: where amyloid starts and grows fastest
    rho    growth rate (per year), eps: spontaneous seeding rate (per year)

The typical pattern follows the published order of amyloid staging (medial parietal and medial
frontal first, lateral association cortex next, occipital and sensorimotor last). The numbers are
illustrative, not fitted to patients. This simulator is the "ground truth" for the proof of concept,
so anything learned from it reflects these assumptions, not real biology.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .phantom import REGIONS, N_REGIONS

TYPICAL = {
    "precuneus_pcc": 1.00,
    "medial_frontal": 0.90,
    "parietal_lat": 0.75,
    "frontal_lat": 0.70,
    "temporal_lat": 0.55,
    "occipital_lat": 0.35,
    "medial_occipital": 0.30,
    "sensorimotor": 0.25,
}

# Used only for the stress test: an onset pattern the model never sees in training.
ATYPICAL = {
    "temporal_lat": 1.00,
    "occipital_lat": 0.80,
    "medial_occipital": 0.70,
    "parietal_lat": 0.55,
    "precuneus_pcc": 0.40,
    "frontal_lat": 0.35,
    "medial_frontal": 0.30,
    "sensorimotor": 0.25,
}

PATTERNS = {"typical": TYPICAL, "atypical": ATYPICAL}


def susceptibility_map(labels: np.ndarray, gm: np.ndarray, pattern: dict, rng: np.random.Generator,
                       jitter: float = 0.15) -> np.ndarray:
    s = np.zeros(labels.shape, dtype=np.float64)
    for k, name in enumerate(REGIONS):
        for lab in (k + 1, k + 1 + N_REGIONS):
            s[labels == lab] = pattern[name] * rng.lognormal(0.0, jitter)
    # Smooth across region borders, renormalized so gray matter edges are not darkened.
    gmf = gm.astype(np.float64)
    num = gaussian_filter(s * gmf, 1.0)
    den = gaussian_filter(gmf, 1.0)
    out = np.where(gm, num / np.maximum(den, 1e-6), 0.0)
    return out


def _neighbor_masks(gm: np.ndarray):
    masks = []
    for axis in range(3):
        for shift in (1, -1):
            masks.append((axis, shift, np.roll(gm, shift, axis=axis) & gm))
    return masks


def simulate(gm: np.ndarray, s: np.ndarray, rho: float, eps: float, D: float,
             times: list[float], dt: float = 0.1) -> list[np.ndarray]:
    """Integrate from c=0 at t=0 and return snapshots at the requested times (years)."""
    idx = np.argsort(np.asarray(times, dtype=float))
    sorted_times = [float(times[i]) for i in idx]
    c = np.zeros(gm.shape, dtype=np.float64)
    nbrs = _neighbor_masks(gm)
    t = 0.0
    snaps = []
    for target in sorted_times:
        n_steps = int(round((target - t) / dt))
        for _ in range(n_steps):
            lap = np.zeros_like(c)
            for axis, shift, m in nbrs:
                lap += np.where(m, np.roll(c, shift, axis=axis) - c, 0.0)
            c = c + dt * (D * lap + rho * s * c * (1.0 - c) + eps * s * (1.0 - c))
            c = np.clip(c, 0.0, 1.0)
            c[~gm] = 0.0
        t += n_steps * dt
        snaps.append(c.copy())
    out: list[np.ndarray] = [None] * len(snaps)  # type: ignore[list-item]
    for j, i in enumerate(idx):
        out[i] = snaps[j]
    return out


def shape_pattern(pattern: dict, gamma: float = 1.0) -> dict:
    """Sensitivity analysis: raise regional susceptibilities to a power. gamma < 1 flattens the
    gradient between early and late regions, gamma > 1 steepens it; gamma = 1 leaves it unchanged."""
    return {k: float(v ** gamma) for k, v in pattern.items()}


def sample_params(rng: np.random.Generator, rho_sd: float = 0.25) -> dict:
    """rho_sd: between-subject log-SD of the growth rate (0.25 in the main cohorts)."""
    resistant = rng.random() < 0.2  # some people stay amyloid-negative
    return {
        "rho": float(rng.lognormal(np.log(0.30), rho_sd)),
        "eps": float(rng.lognormal(np.log(0.004), 0.3) * (0.1 if resistant else 1.0)),
        "D": float(rng.uniform(0.1, 0.4)),
        "resistant": bool(resistant),
    }
