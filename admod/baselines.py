"""Classical baselines. All regional models are fit on training subjects only.

Regional models predict each region's change from baseline to follow-up as two parts:

    change = fix + gap * rate

    fix   does not depend on the forecast gap. It captures regression to the mean: a baseline scan
          that reads high because of noise tends to be followed by a lower measurement.
    rate  yearly change.

Keeping the two apart matters: folding the noise correction into the rate makes it grow with the
gap and ruins long-range forecasts.

copy_last            follow-up = baseline (no change)
RegionalTrend        1 scan.  fix: [1, x0];          rate: [1, x0, x0^2]  (rate-versus-level)
RegionalLinear2      2 scans. fix: [1, x0, x0-x_-1]; rate: [1, x0, x0^2, own slope]
SubjectSlope         2 scans. extend the subject's own prior-to-baseline slope (no fitting)
"""
from __future__ import annotations

import numpy as np

from .phantom import cortical_labels


def copy_last(baseline: np.ndarray) -> np.ndarray:
    return baseline.copy()


def region_means(vol: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(B,N,N,N) -> (B,R) mean per cortical region (NaN if absent)."""
    labs = cortical_labels()
    out = np.full((len(vol), len(labs)), np.nan)
    for i in range(len(vol)):
        for j, lab in enumerate(labs):
            m = labels[i] == lab
            if m.any():
                out[i, j] = vol[i][m].mean()
    return out


def paint(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(B,R) region values -> (B,N,N,N) map, 0 outside cortex."""
    out = np.zeros(labels.shape, np.float32)
    for i in range(len(labels)):
        for j, lab in enumerate(cortical_labels()):
            out[i][labels[i] == lab] = values[i, j]
    return out


def _stack(cols, n):
    return np.stack(cols, 1) if cols else np.zeros((n, 0))


class _RegionalChange:
    n_scans = 1

    def _fix_features(self, b, p):
        raise NotImplementedError

    def _rate_features(self, b, p, g0):
        raise NotImplementedError

    def _inputs(self, cohort):
        rb = region_means(cohort["baseline"], cohort["labels"])
        rp = region_means(cohort["prior"], cohort["labels"]) if self.n_scans == 2 else None
        return rb, rp, cohort.get("dt_prior")

    def fit(self, cohort: dict):
        rb, rp, g0 = self._inputs(cohort)
        rf = region_means(cohort["followup"], cohort["labels"])
        g1 = cohort["dt"]
        self.coefs = []
        for j in range(rb.shape[1]):
            p = None if rp is None else rp[:, j]
            Xf = _stack(self._fix_features(rb[:, j], p), len(rb))
            Xr = np.stack(self._rate_features(rb[:, j], p, g0), 1) * g1[:, None]
            X = np.concatenate([Xf, Xr], 1)
            y = rf[:, j] - rb[:, j]
            ok = np.isfinite(X).all(1) & np.isfinite(y)
            self.coefs.append(np.linalg.lstsq(X[ok], y[ok], rcond=None)[0])
        return self

    def region_parts(self, cohort: dict) -> tuple[np.ndarray, np.ndarray]:
        """(B,R) gap-independent fix and (B,R) yearly rate."""
        if not getattr(self, "coefs", None):
            raise RuntimeError("fit() must be called on training data first")
        rb, rp, g0 = self._inputs(cohort)
        fix, rate = np.zeros_like(rb), np.zeros_like(rb)
        for j, cf in enumerate(self.coefs):
            p = None if rp is None else rp[:, j]
            Xf = _stack(self._fix_features(rb[:, j], p), len(rb))
            Xr = np.stack(self._rate_features(rb[:, j], p, g0), 1)
            fix[:, j] = Xf @ cf[:Xf.shape[1]]
            rate[:, j] = Xr @ cf[Xf.shape[1]:]
        return np.nan_to_num(fix), np.nan_to_num(rate)

    def maps(self, cohort: dict) -> tuple[np.ndarray, np.ndarray]:
        fix, rate = self.region_parts(cohort)
        return paint(fix, cohort["labels"]), paint(rate, cohort["labels"])

    def rate_map(self, cohort: dict) -> np.ndarray:
        return self.maps(cohort)[1]

    def predict(self, cohort: dict) -> np.ndarray:
        """Uses baseline (and prior) scans only. The follow-up scan is never read."""
        fix, rate = self.maps(cohort)
        return cohort["baseline"] + fix + cohort["dt"][:, None, None, None] * rate

    def state(self) -> dict:
        return {"cls": type(self).__name__, "coefs": [c.tolist() for c in self.coefs]}


class RegionalTrend(_RegionalChange):
    n_scans = 1

    def _fix_features(self, b, p):
        return [np.ones_like(b), b]

    def _rate_features(self, b, p, g0):
        return [np.ones_like(b), b, b ** 2]


class RegionalLinear2(_RegionalChange):
    n_scans = 2

    def _fix_features(self, b, p):
        return [np.ones_like(b), b, b - p]

    def _rate_features(self, b, p, g0):
        return [np.ones_like(b), b, b ** 2, (b - p) / g0]


class RegionalRateOnly2(_RegionalChange):
    """2 scans, rate term only (no gap-independent term)."""
    n_scans = 2

    def _fix_features(self, b, p):
        return []

    def _rate_features(self, b, p, g0):
        return [np.ones_like(b), b, b ** 2, (b - p) / g0]


class RegionalLinear2WM(RegionalLinear2):
    """Regional linear (2 scans) + deep white-matter SUVR of both scans in the gap-independent term."""

    def _inputs(self, cohort):
        rb, rp, g0 = super()._inputs(cohort)
        self._w0 = deep_wm_means(cohort["baseline"], cohort["labels"])
        self._wp = deep_wm_means(cohort["prior"], cohort["labels"])
        return rb, rp, g0

    def _fix_features(self, b, p):
        return [np.ones_like(b), b, b - p, self._w0, self._w0 - self._wp]


class RegionalCross2:
    """2 scans. Same phi + g1*psi split as RegionalLinear2, but phi sees every cortical region of both
    scans and deep white matter: [1, all regional x0, all regional (x0 - x_-1), wm0, wm0 - wm_-1]
    (35 features). psi is the own-region rate [1, x0, x0^2, own slope]. Ridge regression on standardized
    features (intercept not penalized); lambda is chosen on the validation cohort only."""
    n_scans = 2
    LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    # IEEE Access revision: the medRxiv v2 choice (0.01) sat at the grid edge, so the grid is extended down.
    LAMBDAS_EXT = (1e-4, 1e-3, 3e-3) + LAMBDAS

    def __init__(self, lam: float | None = None, grid: tuple | None = None):
        self.lam = lam
        if grid is not None:
            self.LAMBDAS = tuple(grid)

    @staticmethod
    def _prep(c):
        rb, rp = region_means(c["baseline"], c["labels"]), region_means(c["prior"], c["labels"])
        w0, wp = deep_wm_means(c["baseline"], c["labels"]), deep_wm_means(c["prior"], c["labels"])
        return rb, rp, w0, wp

    @staticmethod
    def _X(rb, rp, w0, wp, g0, g1, j):
        n = len(rb)
        fix = np.concatenate([np.ones((n, 1)), rb, rb - rp, w0[:, None], (w0 - wp)[:, None]], 1)
        b, p = rb[:, j], rp[:, j]
        rate = np.stack([np.ones_like(b), b, b ** 2, (b - p) / g0], 1)
        return fix, rate, np.concatenate([fix, rate * g1[:, None]], 1)

    def _fit_lam(self, P, y_all, g0, g1, lam):
        rb, rp, w0, wp = P
        W = []
        for j in range(rb.shape[1]):
            _, _, X = self._X(rb, rp, w0, wp, g0, g1, j)
            mu, sd = X.mean(0), X.std(0)
            const = sd == 0
            sd[const], mu[const] = 1.0, 0.0
            Xs = (X - mu) / sd
            pen = lam * np.eye(Xs.shape[1])
            pen[const, const] = 0.0
            w = np.linalg.solve(Xs.T @ Xs + pen, Xs.T @ y_all[:, j])
            W.append((w, mu, sd))
        return W

    def fit(self, cohort: dict, val: dict | None = None):
        P = self._prep(cohort)
        y = region_means(cohort["followup"], cohort["labels"]) - P[0]
        g0, g1 = cohort["dt_prior"].astype(float), cohort["dt"].astype(float)
        if self.lam is None:
            if val is None:
                raise ValueError("lambda must be given or chosen on a validation cohort")
            Pv = self._prep(val)
            yv = region_means(val["followup"], val["labels"]) - Pv[0]
            def score(lam):
                self.W = self._fit_lam(P, y, g0, g1, lam)
                pv = self._region_change(Pv, val["dt_prior"].astype(float), val["dt"].astype(float))
                t = yv.ravel()
                return 1 - ((pv.ravel() - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            self.val_scores = {lam: float(score(lam)) for lam in self.LAMBDAS}
            self.lam = max(self.val_scores, key=self.val_scores.get)
        self.W = self._fit_lam(P, y, g0, g1, self.lam)
        return self

    def _region_change(self, P, g0, g1):
        rb, rp, w0, wp = P
        out = np.zeros_like(rb)
        for j, (w, mu, sd) in enumerate(self.W):
            _, _, X = self._X(rb, rp, w0, wp, g0, g1, j)
            out[:, j] = ((X - mu) / sd) @ w
        return out

    def predict(self, cohort: dict) -> np.ndarray:
        """Uses baseline and prior scans only. The follow-up scan is never read."""
        P = self._prep(cohort)
        chg = self._region_change(P, cohort["dt_prior"].astype(float), cohort["dt"].astype(float))
        return cohort["baseline"] + paint(np.nan_to_num(chg), cohort["labels"])


def deep_wm_means(vol: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    from .phantom import WHITE_MATTER
    return np.array([vol[i][binary_erosion(labels[i] == WHITE_MATTER, iterations=1)].mean() for i in range(len(vol))])


class SubjectSlope:
    n_scans = 2

    def fit(self, cohort: dict):
        return self

    def region_parts(self, cohort: dict):
        rb = region_means(cohort["baseline"], cohort["labels"])
        rp = region_means(cohort["prior"], cohort["labels"])
        return np.zeros_like(rb), np.nan_to_num((rb - rp) / cohort["dt_prior"][:, None])

    def maps(self, cohort: dict):
        fix, rate = self.region_parts(cohort)
        return paint(fix, cohort["labels"]), paint(rate, cohort["labels"])

    def rate_map(self, cohort: dict) -> np.ndarray:
        return self.maps(cohort)[1]

    def predict(self, cohort: dict) -> np.ndarray:
        return cohort["baseline"] + cohort["dt"][:, None, None, None] * self.rate_map(cohort)

    def state(self) -> dict:
        return {"cls": "SubjectSlope", "coefs": []}


REGISTRY = {"RegionalTrend": RegionalTrend, "RegionalLinear2": RegionalLinear2, "RegionalRateOnly2": RegionalRateOnly2,
            "RegionalLinear2WM": RegionalLinear2WM, "SubjectSlope": SubjectSlope}


def from_state(st: dict):
    obj = REGISTRY[st["cls"]]()
    obj.coefs = [np.array(c) for c in st["coefs"]]
    return obj
