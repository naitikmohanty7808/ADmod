"""Tests for splits, baselines, metrics, and the ADmod network."""
import os

import numpy as np
import pytest
import torch

from admod.baselines import RegionalLinear2, RegionalTrend, SubjectSlope, copy_last, paint, region_means
from admod.dataset import split_ids
from admod.metrics import per_subject, scores, summarize
from admod.models import ADmod, build_features, loss_weights, n_features
from admod.phantom import cortical_labels, make_brain

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------- splits: no person may appear in two splits ----------

def test_split_disjoint_and_complete():
    sp = split_ids(np.arange(100), seed=5)
    tr, va, te = (set(sp[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(range(100))


def test_split_deterministic():
    a, b = split_ids(np.arange(50), seed=1), split_ids(np.arange(50), seed=1)
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "train.npz")), reason="data not generated")
def test_saved_splits_disjoint_and_flagged_synthetic():
    ids = {}
    for name in ("train", "val", "test", "stress_atypical", "stress_long_gap"):
        with np.load(os.path.join(DATA, f"{name}.npz")) as z:
            ids[name] = set(z["ids"].tolist())
            assert bool(z["synthetic"])
    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not ids[names[i]] & ids[names[j]], (names[i], names[j])


# ---------- helpers ----------

def _toy(n=8, seed=0, region_constant=False):
    rng = np.random.default_rng(seed)
    labels = np.stack([make_brain(rng, 32)["labels"] for _ in range(n)])
    if region_constant:
        rb = rng.uniform(1.0, 2.0, size=(n, 16))
        base = paint(rb, labels)
    else:
        base = rng.uniform(1.0, 2.0, size=labels.shape).astype(np.float32)
    return {"labels": labels, "baseline": base.astype(np.float32),
            "dt": rng.uniform(1, 4, size=n).astype(np.float32),
            "dt_prior": rng.uniform(1.5, 3, size=n).astype(np.float32)}


def _ctx(labels):
    return (labels >= 1) & (labels <= 16)


# ---------- metrics ----------

def test_perfect_prediction_scores_one():
    c = _toy()
    target = c["baseline"] + 0.05 * _ctx(c["labels"]) * np.random.default_rng(1).random(c["baseline"].shape)
    s = summarize(target, target, c["baseline"], c["labels"])
    assert s["voxel_r2"] == pytest.approx(1.0) and s["regional_r2"] == pytest.approx(1.0)
    assert s["composite_mae"] == pytest.approx(0.0)


def test_copy_last_scores_below_zero_when_amyloid_rises():
    c = _toy()
    target = c["baseline"] + _ctx(c["labels"]) * (0.05 + 0.02 * np.random.default_rng(2).random(c["baseline"].shape))
    s = summarize(copy_last(c["baseline"]), target, c["baseline"], c["labels"])
    assert s["voxel_r2"] < 0 and s["regional_r2"] < 0


def test_predicting_average_change_scores_about_zero():
    c = _toy()
    ctx = _ctx(c["labels"])
    target = c["baseline"] + ctx * np.random.default_rng(3).normal(0.05, 0.02, c["baseline"].shape)
    pred = c["baseline"] + ctx * (target - c["baseline"])[ctx].mean()
    assert abs(scores(per_subject(pred, target, c["baseline"], c["labels"]))["voxel_r2"]) < 0.01


# ---------- baselines ----------

def test_region_means_and_paint_roundtrip():
    c = _toy(n=3, region_constant=True)
    rm = region_means(c["baseline"], c["labels"])
    assert np.allclose(paint(rm, c["labels"])[_ctx(c["labels"])], c["baseline"][_ctx(c["labels"])], atol=1e-6)


def test_regional_trend_recovers_known_rule():
    c = _toy(n=14, region_constant=True)
    rate = lambda x: 0.01 + 0.02 * x - 0.004 * x ** 2
    c["followup"] = c["baseline"] + c["dt"][:, None, None, None] * rate(c["baseline"]) * _ctx(c["labels"])
    tr = {k: v[:10] for k, v in c.items()}
    te = {k: v[10:] for k, v in c.items()}
    pred = RegionalTrend().fit(tr).predict(te)
    assert np.allclose(pred, te["followup"], atol=1e-4)


def test_regional_linear2_recovers_rule_with_own_slope():
    c = _toy(n=24, region_constant=True)
    rng = np.random.default_rng(5)
    own = rng.uniform(0.0, 0.05, size=(24, 16))
    c["prior"] = c["baseline"] - c["dt_prior"][:, None, None, None] * paint(own, c["labels"])
    true_rate = 0.01 + 0.005 * c["baseline"] + 0.6 * paint(own, c["labels"])
    c["followup"] = c["baseline"] + c["dt"][:, None, None, None] * true_rate * _ctx(c["labels"])
    tr = {k: v[:18] for k, v in c.items()}
    te = {k: v[18:] for k, v in c.items()}
    pred = RegionalLinear2().fit(tr).predict(te)
    assert np.allclose(pred, te["followup"], atol=1e-4)


def test_subject_slope_extends_own_line():
    c = _toy(n=4, region_constant=True)
    own = np.random.default_rng(6).uniform(0, 0.05, size=(4, 16))
    c["prior"] = c["baseline"] - c["dt_prior"][:, None, None, None] * paint(own, c["labels"])
    expect = c["baseline"] + c["dt"][:, None, None, None] * paint(own, c["labels"])
    assert np.allclose(SubjectSlope().predict(c), expect, atol=1e-5)


def test_baselines_never_read_followup():
    c = _toy(n=12, region_constant=True)
    c["prior"] = c["baseline"] - 0.01
    c["followup"] = c["baseline"] + 0.05 * _ctx(c["labels"])
    rt, rl = RegionalTrend().fit(c), RegionalLinear2().fit(c)
    blind = {k: v for k, v in c.items() if k != "followup"}
    for m in (rt, rl, SubjectSlope()):
        m.predict(blind)  # would raise KeyError if the follow-up were read


def test_noise_correction_does_not_grow_with_gap():
    """A pure regression-to-the-mean effect must land in the gap-independent part."""
    c = _toy(n=40, region_constant=True)
    rng = np.random.default_rng(8)
    true = paint(rng.uniform(1.0, 2.0, size=(40, 16)), c["labels"])
    noise = paint(rng.normal(0, 0.05, size=(40, 16)), c["labels"])
    c["baseline"] = true + noise
    c["prior"] = true + paint(rng.normal(0, 0.05, size=(40, 16)), c["labels"])
    c["followup"] = true  # no real change; measured baseline noise only
    fix, rate = RegionalLinear2().fit(c).region_parts(c)
    # restored test (IEEE Access revision): with no real change, the gap-independent correction tracks
    # the regression-to-the-mean term (the original assertion on the rate was lost with the repository)
    rb = region_means(c["baseline"], c["labels"]); rt = region_means(true, c["labels"])
    assert np.corrcoef(fix.ravel(), (rt - rb).ravel())[0, 1] > 0.5


def test_unfitted_regional_model_raises():
    with pytest.raises(RuntimeError):
        RegionalTrend().predict(_toy(n=1))


# ---------- ADmod network ----------

def _toy_model_inputs(n_scans, n=2):
    c = _toy(n=n, seed=9)
    c["prior"] = c["baseline"] - 0.02
    f = torch.from_numpy(build_features(c, n_scans))
    pri = torch.stack([torch.full((n, 32, 32, 32), -0.01), torch.full((n, 32, 32, 32), 0.03)], 1)
    return c, f, torch.from_numpy(c["dt"]), pri


@pytest.mark.parametrize("n_scans", [1, 2])
def test_feature_shapes(n_scans):
    c, f, d, p = _toy_model_inputs(n_scans)
    assert f.shape == (2, n_features(n_scans), 32, 32, 32)
    assert torch.allclose(f[:, 0], torch.from_numpy(c["baseline"]))


@pytest.mark.parametrize("n_scans", [1, 2])
def test_untrained_admod_equals_its_prior(n_scans):
    c, f, d, p = _toy_model_inputs(n_scans)
    pred, _ = ADmod(n_scans=n_scans, width=8)(f, d, p)
    expect = torch.from_numpy(c["baseline"])[:, None] + p[:, 0:1] + d.view(-1, 1, 1, 1, 1) * p[:, 1:2]
    assert torch.allclose(pred, expect, atol=1e-6)


def test_learned_terms_zero_outside_brain():
    c, f, d, p = _toy_model_inputs(2)
    m = ADmod(n_scans=2, width=8)
    for h in (m.head_rate, m.head_fix):
        torch.nn.init.normal_(h.weight, std=0.5)
        torch.nn.init.normal_(h.bias, std=0.5)
    m.eval()
    pred, _ = m(f, d, torch.zeros_like(p))
    outside = torch.from_numpy(c["labels"] == 0)
    assert torch.allclose(pred[:, 0][outside], torch.from_numpy(c["baseline"])[outside])


def test_admod_can_learn_a_simple_change():
    """Overfit check: learn 'cortex rises 0.03 per year' on two subjects with no prior."""
    torch.manual_seed(0)
    c, f, d, _ = _toy_model_inputs(2)
    ctx = _ctx(c["labels"]).astype(np.float32)
    y = torch.from_numpy(c["baseline"] + c["dt"][:, None, None, None] * 0.03 * ctx)[:, None]
    w = torch.from_numpy(loss_weights(c["labels"]))
    m = ADmod(n_scans=2, width=8)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    zero = torch.zeros(2, 2, 32, 32, 32)
    losses = []
    for _ in range(60):
        pred, _ = m(f, d, zero)
        loss = (w * (pred - y) ** 2).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
