"""Synthetic brain anatomy ("phantom") generator.

SYNTHETIC DATA ONLY. These are geometric stand-ins for a brain, not real anatomy.
Each subject gets a slightly different brain shape so the model cannot memorize one geometry.

Axes: 0 = left/right (x), 1 = back/front (y), 2 = bottom/top (z).
"""
from __future__ import annotations

import numpy as np

# Cortical regions, ordered. Label for left hemisphere = index + 1, right = index + 1 + len(REGIONS).
REGIONS = [
    "frontal_lat",
    "sensorimotor",
    "parietal_lat",
    "temporal_lat",
    "occipital_lat",
    "precuneus_pcc",
    "medial_frontal",
    "medial_occipital",
]
N_REGIONS = len(REGIONS)

# Non-cortical tissue labels
BACKGROUND = 0
CEREB_GM = 50   # cerebellar gray matter: the reference region (no amyloid in this simulator)
CEREB_WM = 51
WHITE_MATTER = 60
CSF = 70


def region_label(name: str, hemi: str) -> int:
    k = REGIONS.index(name)
    return k + 1 if hemi == "L" else k + 1 + N_REGIONS


def cortical_labels() -> list[int]:
    return list(range(1, 2 * N_REGIONS + 1))


def region_name(label: int) -> str:
    if 1 <= label <= 2 * N_REGIONS:
        k = (label - 1) % N_REGIONS
        hemi = "L" if label <= N_REGIONS else "R"
        return f"{REGIONS[k]}_{hemi}"
    return {CEREB_GM: "cerebellum_gm", CEREB_WM: "cerebellum_wm", WHITE_MATTER: "white_matter",
            CSF: "csf", BACKGROUND: "background"}.get(label, f"label_{label}")


def _grid(n: int):
    ax = (np.arange(n) - (n - 1) / 2) / (n / 2)
    return np.meshgrid(ax, ax, ax, indexing="ij")


def _smooth_warp(rng: np.random.Generator, x, y, z, amp: float = 0.03):
    """Low-frequency random displacement so every subject has a different shape."""
    out = []
    for _ in range(3):
        d = np.zeros_like(x)
        for _ in range(2):
            f = rng.uniform(0.5, 1.5, size=3) * np.pi
            ph = rng.uniform(0, 2 * np.pi, size=3)
            d += amp * np.sin(f[0] * x + ph[0]) * np.sin(f[1] * y + ph[1]) * np.sin(f[2] * z + ph[2])
        out.append(d)
    return x + out[0], y + out[1], z + out[2]


def make_brain(rng: np.random.Generator, n: int = 32) -> dict:
    """Return a dict with an int16 label volume and convenience masks for one synthetic subject."""
    x, y, z = _grid(n)
    x, y, z = _smooth_warp(rng, x, y, z)

    s = rng.uniform(0.93, 1.07, size=3)  # per-axis head-size variation
    c_ctr = np.array([0.0, 0.06, 0.12])
    c_rad = np.array([0.66, 0.82, 0.58]) * s
    d = np.sqrt(((x - c_ctr[0]) / c_rad[0]) ** 2 + ((y - c_ctr[1]) / c_rad[1]) ** 2
                + ((z - c_ctr[2]) / c_rad[2]) ** 2)

    cb_ctr = np.array([0.0, -0.52, -0.52])
    cb_rad = np.array([0.42, 0.24, 0.20]) * s
    dcb = np.sqrt(((x - cb_ctr[0]) / cb_rad[0]) ** 2 + ((y - cb_ctr[1]) / cb_rad[1]) ** 2
                  + ((z - cb_ctr[2]) / cb_rad[2]) ** 2)
    cerebellum = dcb < 1.0
    cerebrum = (d < 1.0) & ~(dcb < 1.25)  # small gap between cerebrum and cerebellum

    vent = np.zeros_like(cerebrum)
    for sx in (-1, 1):
        dv = np.sqrt(((x - sx * 0.11) / 0.07) ** 2 + ((y - 0.02) / 0.30) ** 2 + ((z - 0.14) / 0.12) ** 2)
        vent |= dv < 1.0
    vent &= cerebrum

    shell = cerebrum & (d >= 0.80)
    medial_wall = cerebrum & (np.abs(x) < 0.07) & (d > 0.5)
    gm = (shell | medial_wall) & ~vent

    labels = np.zeros(x.shape, dtype=np.int16)
    labels[cerebrum] = WHITE_MATTER
    labels[vent] = CSF

    ny = (y - c_ctr[1]) / c_rad[1]
    nz = (z - c_ctr[2]) / c_rad[2]
    medial = np.abs(x) < 0.12
    name = np.empty(x.shape, dtype=object)
    name[:] = ""
    lat = ~medial
    name[medial & (ny < -0.55)] = "medial_occipital"
    name[medial & (ny >= -0.55) & (ny < 0.05)] = "precuneus_pcc"
    name[medial & (ny >= 0.05)] = "medial_frontal"
    name[lat & (ny < -0.6)] = "occipital_lat"
    temporal = lat & (ny >= -0.6) & (nz < -0.15) & (ny < 0.45)
    name[temporal] = "temporal_lat"
    rest = lat & (ny >= -0.6) & ~temporal
    name[rest & (ny > 0.25)] = "frontal_lat"
    name[rest & (ny > 0.0) & (ny <= 0.25)] = "sensorimotor"
    name[rest & (ny <= 0.0)] = "parietal_lat"

    for k, rname in enumerate(REGIONS):
        sel = gm & (name == rname)
        labels[sel & (x < 0)] = k + 1
        labels[sel & (x >= 0)] = k + 1 + N_REGIONS

    labels[cerebellum] = CEREB_WM
    labels[cerebellum & (dcb >= 0.55)] = CEREB_GM

    return {
        "labels": labels,
        "brain": labels > 0,
        "cortex_gm": (labels >= 1) & (labels <= 2 * N_REGIONS),
        "cereb_gm": labels == CEREB_GM,
    }
