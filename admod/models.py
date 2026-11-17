"""ADmod: hybrid residual 3D encoder-decoder for forecasting a follow-up amyloid PET scan.

    forecast = baseline + (prior_fix + gap * prior_rate) + fix + gap * rate

    prior_fix, prior_rate  the two parts of a regional statistical model (fit on training subjects
                only): a gap-independent correction and a yearly change; zero in the no-prior ablation
    fix         learned voxel correction that does not scale with the gap (removes baseline-scan
                noise and misregistration)
    rate        learned voxel correction to the yearly change

Input channels (n_scans = 2): baseline SUVR, prior SUVR, cortical gray matter mask, brain mask,
forecast gap / 4, prior gap / 4, region-mean maps of baseline and prior, x/y/z template position,
prior_fix, prior_rate. With n_scans = 1 the prior-scan channels are dropped.

Both output layers start at zero, so before training ADmod equals its regional prior
(or "copy the last scan" without a prior). Any gain over the prior must be learned.

Architecture flags (IEEE Access revision; defaults reproduce the medRxiv v2 model):
    gap_input    forecast-gap and prior-gap channels, and the gap input of the global MLP
    global_ctx   pooled-bottleneck MLP added back to the bottleneck
    region_maps  region-mean maps of the input scans
    coords       x/y/z template position channels
    head         "two": baseline + fix + gap * rate;  "direct": baseline + one learned correction
    prior_ch     the two regional-prior channels (all zero when no prior is used)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .baselines import paint, region_means
from .phantom import N_REGIONS


def n_features(n_scans: int) -> int:
    return 8 if n_scans == 2 else 5


def brain_index(n_scans: int) -> int:
    return 3 if n_scans == 2 else 2


def build_features(cohort: dict, n_scans: int, idx=None) -> np.ndarray:
    """(B, C, N, N, N) float32 features without position and prior channels."""
    sl = slice(None) if idx is None else idx
    b, lab, g1 = cohort["baseline"][sl], cohort["labels"][sl], cohort["dt"][sl]
    gm = ((lab >= 1) & (lab <= 2 * N_REGIONS)).astype(np.float32)
    brain = (lab > 0).astype(np.float32)
    ones = np.ones(b.shape, np.float32)
    rb = paint(region_means(b, lab), lab)
    if n_scans == 2:
        p, g0 = cohort["prior"][sl], cohort["dt_prior"][sl]
        rp = paint(region_means(p, lab), lab)
        chans = [b, p, gm, brain, ones * (g1 / 4.0)[:, None, None, None],
                 ones * (g0 / 4.0)[:, None, None, None], rb, rp]
    else:
        chans = [b, gm, brain, ones * (g1 / 4.0)[:, None, None, None], rb]
    return np.ascontiguousarray(np.stack(chans, 1).astype(np.float32))


def _block(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    # Batch norm, not per-sample norms: group/instance norm remove each scan's overall signal
    # level, which the accumulation rate depends on.
    return nn.Sequential(
        nn.Conv3d(cin, cout, k, padding=k // 2), nn.BatchNorm3d(cout), nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.SiLU(),
    )


ARCH_DEFAULT = {"gap_input": True, "global_ctx": True, "region_maps": True, "coords": True,
                "head": "two", "prior_ch": True}


def feature_index(n_scans: int, gap_input: bool, region_maps: bool) -> list[int]:
    """Indices of build_features channels the model reads."""
    if n_scans == 2:  # [b, p, gm, brain, g1/4, g0/4, rb, rp]
        keep = [0, 1, 2, 3] + ([4, 5] if gap_input else []) + ([6, 7] if region_maps else [])
    else:             # [b, gm, brain, g1/4, rb]
        keep = [0, 1, 2] + ([3] if gap_input else []) + ([4] if region_maps else [])
    return keep


class ADmod(nn.Module):
    def __init__(self, n_scans: int = 2, width: int = 12, grid: int = 32, arch: dict | None = None):
        super().__init__()
        a = dict(ARCH_DEFAULT, **(arch or {}))
        assert a["head"] in ("two", "direct")
        self.arch = a
        self.n_scans = n_scans
        self.brain_idx = brain_index(n_scans)
        self.keep = feature_index(n_scans, a["gap_input"], a["region_maps"])
        self.subset = len(self.keep) != n_features(n_scans)
        c = width
        cin = len(self.keep) + (3 if a["coords"] else 0) + (2 if a["prior_ch"] else 0)
        ax = (torch.arange(grid, dtype=torch.float32) - (grid - 1) / 2) / (grid / 2)
        self.register_buffer("coords", torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"))[None])
        self.enc1 = _block(cin, c, k=5)  # wider first kernel for context (5 not 7: 1.7x faster on CPU)
        self.enc2 = _block(c, 2 * c)
        self.enc3 = _block(2 * c, 4 * c)
        self.pool = nn.MaxPool3d(2)
        if a["global_ctx"]:
            self.glob = nn.Sequential(nn.Linear(4 * c + 1, 4 * c), nn.SiLU(), nn.Linear(4 * c, 4 * c))
        self.up2 = nn.ConvTranspose3d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = _block(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose3d(2 * c, c, 2, stride=2)
        self.dec1 = _block(2 * c, c)
        self.head_fix = nn.Conv3d(c, 1, 1)
        heads = [self.head_fix]
        if a["head"] == "two":
            self.head_rate = nn.Conv3d(c, 1, 1)
            heads.append(self.head_rate)
        for h in heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, feat: torch.Tensor, dt: torch.Tensor, prior: torch.Tensor):
        """feat: (B,C,N,N,N), dt: (B,) years, prior: (B,2,N,N,N) [fix, yearly rate]. Returns (forecast, rate)."""
        B, A = feat.shape[0], self.arch
        parts = [feat[:, self.keep] if self.subset else feat]
        if A["coords"]:
            parts.append(self.coords.expand(B, -1, -1, -1, -1))
        if A["prior_ch"]:
            parts.append(prior)
        x = torch.cat(parts, 1)
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        if A["global_ctx"]:
            gin = dt.view(-1, 1) / 4.0 if A["gap_input"] else torch.zeros_like(dt.view(-1, 1))
            g = self.glob(torch.cat([c.mean(dim=(2, 3, 4)), gin], 1))
            c = c + g[:, :, None, None, None]
        d = self.dec2(torch.cat([self.up2(c), b], 1))
        e = self.dec1(torch.cat([self.up1(d), a], 1))
        brain = feat[:, self.brain_idx:self.brain_idx + 1]
        fix = self.head_fix(e) * brain
        dtv = dt.view(-1, 1, 1, 1, 1)
        base = feat[:, 0:1] + prior[:, 0:1] + dtv * prior[:, 1:2]
        if A["head"] == "direct":
            return base + fix, fix
        rate = self.head_rate(e) * brain
        return base + fix + dtv * rate, rate


def loss_weights(labels: np.ndarray) -> np.ndarray:
    """Brain voxels weight 1, cortical gray matter weight 3 (where amyloid accumulates)."""
    gm = ((labels >= 1) & (labels <= 2 * N_REGIONS)).astype(np.float32)
    return ((labels > 0).astype(np.float32) + 2.0 * gm)[:, None]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
