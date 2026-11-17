"""Amyloid PET image simulator (SYNTHETIC).

Turns an amyloid map into a simulated amyloid PET scan:
  - white matter has high non-specific uptake (as with real amyloid tracers)
  - cortical gray matter signal rises with amyloid load
  - scanner blur (partial volume), voxel noise, injected-dose scaling, small misregistration
  - per-scan variation in target-to-reference uptake (stands in for blood flow and uptake-time
    effects); this term sets the scan-rescan variability of the cortical composite SUVR
  - final image is an SUVR: divided by the cerebellar gray matter mean (standard reference region)

`noise_scale` multiplies every random error term. 1.0 is the calibrated setting.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

from .phantom import CEREB_GM, CEREB_WM, WHITE_MATTER, CSF, N_REGIONS

VOXEL_MM = 5.0
FWHM_MM = 7.0
# Per-scan log-SD of the target-to-reference factor. Calibrated in calibrate.py so the mean absolute
# scan-rescan difference of the cortical composite SUVR matches the value set there.
GLOBAL_SD = 0.018


def sample_tissue(rng: np.random.Generator) -> dict:
    """Per-subject tracer properties, shared by all scans of the same person."""
    return {
        "wm": float(rng.uniform(1.5, 1.9)),
        "gm0": float(rng.uniform(0.95, 1.10)),
        "amp": float(rng.uniform(1.2, 1.6)),   # SUVR added at full amyloid load
        "noise": float(rng.uniform(0.03, 0.05)),
    }


def true_uptake(labels: np.ndarray, c: np.ndarray, tissue: dict, gain: float = 1.0) -> np.ndarray:
    img = np.full(labels.shape, 0.05)
    img[labels == WHITE_MATTER] = tissue["wm"]
    img[labels == CSF] = 0.2
    img[labels == CEREB_WM] = 1.3
    img[labels == CEREB_GM] = 1.0
    ctx = (labels >= 1) & (labels <= 2 * N_REGIONS)
    img[ctx] = tissue["gm0"] + tissue["amp"] * c[ctx]
    cerebrum = (labels >= 1) & (labels <= 2 * N_REGIONS) | (labels == WHITE_MATTER) | (labels == CSF)
    img[cerebrum] *= gain
    return img


def _ref_mean(img: np.ndarray, labels: np.ndarray) -> float:
    return float(img[labels == CEREB_GM].mean())


def scan(labels: np.ndarray, c: np.ndarray, tissue: dict, rng: np.random.Generator,
         noise_scale: float = 1.0, global_sd: float | None = None,
         max_shift_vox: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy SUVR scan, noise-free SUVR scan) as float32 volumes.

    Random draws happen in a fixed order regardless of noise_scale, so the same seed with a
    different noise_scale gives the same subject with different noise.
    """
    gsd = GLOBAL_SD if global_sd is None else global_sd
    sigma = FWHM_MM / 2.3548 / VOXEL_MM
    clean = gaussian_filter(true_uptake(labels, c, tissue), sigma)
    clean_suvr = clean / _ref_mean(clean, labels)

    z_gain = rng.standard_normal()
    scale = rng.uniform(0.7, 1.4)  # injected dose / body weight; cancels in SUVR
    n = gaussian_filter(rng.standard_normal(labels.shape), 0.7)
    n /= n.std()
    shift = rng.uniform(-max_shift_vox, max_shift_vox, size=3)

    gain = float(np.exp(noise_scale * gsd * z_gain))
    blurred = gaussian_filter(true_uptake(labels, c, tissue, gain), sigma)
    noisy = scale * (blurred + noise_scale * tissue["noise"] * np.sqrt(np.maximum(blurred, 0.0)) * n)
    noisy = nd_shift(noisy, noise_scale * shift, order=1, mode="nearest")
    suvr = noisy / _ref_mean(noisy, labels)
    return suvr.astype(np.float32), clean_suvr.astype(np.float32)
