"""Tests for the synthetic brain, amyloid, and PET simulators."""
import numpy as np
import pytest

from admod.phantom import make_brain, cortical_labels, CEREB_GM, REGIONS, region_label
from admod.amyloid import TYPICAL, simulate, susceptibility_map, sample_params
from admod.pet import scan, sample_tissue, VOXEL_MM


@pytest.fixture(scope="module")
def brain():
    return make_brain(np.random.default_rng(7), 32)


def test_phantom_reproducible():
    a = make_brain(np.random.default_rng(3), 32)["labels"]
    b = make_brain(np.random.default_rng(3), 32)["labels"]
    assert np.array_equal(a, b)


def test_phantom_differs_between_subjects():
    a = make_brain(np.random.default_rng(3), 32)["labels"]
    b = make_brain(np.random.default_rng(4), 32)["labels"]
    assert not np.array_equal(a, b)


def test_every_cortical_region_present(brain):
    for lab in cortical_labels():
        assert (brain["labels"] == lab).sum() > 20, f"region {lab} too small"


def test_reference_region_separate_from_cortex(brain):
    assert brain["cereb_gm"].sum() > 100
    assert not (brain["cereb_gm"] & brain["cortex_gm"]).any()


def test_amyloid_bounded_and_confined(brain):
    rng = np.random.default_rng(0)
    gm = brain["cortex_gm"]
    s = susceptibility_map(brain["labels"], gm, TYPICAL, rng)
    (c,) = simulate(gm, s, rho=0.4, eps=0.01, D=0.3, times=[25.0])
    assert c.min() >= 0.0 and c.max() <= 1.0
    assert np.all(c[~gm] == 0.0)


def test_amyloid_only_grows(brain):
    rng = np.random.default_rng(0)
    gm = brain["cortex_gm"]
    s = susceptibility_map(brain["labels"], gm, TYPICAL, rng)
    c1, c2, c3 = simulate(gm, s, 0.3, 0.004, 0.2, [5.0, 10.0, 20.0])
    assert c1.sum() < c2.sum() < c3.sum()


def test_snapshots_returned_in_requested_order(brain):
    rng = np.random.default_rng(0)
    gm = brain["cortex_gm"]
    s = susceptibility_map(brain["labels"], gm, TYPICAL, rng)
    a = simulate(gm, s, 0.3, 0.004, 0.2, [12.0, 4.0])
    b = simulate(gm, s, 0.3, 0.004, 0.2, [4.0, 12.0])
    assert np.allclose(a[0], b[1]) and np.allclose(a[1], b[0])


def test_typical_staging_order(brain):
    """Medial parietal (precuneus) should fill before sensorimotor cortex."""
    rng = np.random.default_rng(1)
    gm = brain["cortex_gm"]
    s = susceptibility_map(brain["labels"], gm, TYPICAL, rng, jitter=0.0)
    (c,) = simulate(gm, s, 0.3, 0.004, 0.2, [12.0])
    L = brain["labels"]
    pre = c[(L == region_label("precuneus_pcc", "L")) | (L == region_label("precuneus_pcc", "R"))].mean()
    sm = c[(L == region_label("sensorimotor", "L")) | (L == region_label("sensorimotor", "R"))].mean()
    assert pre > 2 * sm


def test_suvr_reference_region_is_one(brain):
    rng = np.random.default_rng(0)
    c = np.zeros(brain["labels"].shape)
    suvr, clean = scan(brain["labels"], c, sample_tissue(rng), rng)
    assert abs(suvr[brain["cereb_gm"]].mean() - 1.0) < 1e-4
    assert abs(clean[brain["cereb_gm"]].mean() - 1.0) < 1e-4


def test_dose_scaling_cancels_without_noise(brain):
    """With noise and misalignment off, SUVR must not depend on the injected dose."""
    tissue = dict(sample_tissue(np.random.default_rng(0)), noise=0.0)
    c = np.zeros(brain["labels"].shape)
    s1, clean = scan(brain["labels"], c, tissue, np.random.default_rng(1), global_sd=0.0, max_shift_vox=0.0)
    s2, _ = scan(brain["labels"], c, tissue, np.random.default_rng(2), global_sd=0.0, max_shift_vox=0.0)
    assert np.allclose(s1, s2, atol=1e-5) and np.allclose(s1, clean, atol=1e-5)


def test_zero_noise_scale_gives_clean_scan(brain):
    tissue = sample_tissue(np.random.default_rng(0))
    c = brain["cortex_gm"] * 0.5
    noisy, clean = scan(brain["labels"], c, tissue, np.random.default_rng(3), noise_scale=0.0)
    assert np.allclose(noisy, clean, atol=1e-5)


def test_same_seed_different_noise_scale_same_subject():
    from admod.dataset import make_subject
    a = make_subject((123, "typical", (1.0, 4.0), 32, 1.0))
    b = make_subject((123, "typical", (1.0, 4.0), 32, 2.0))
    assert np.array_equal(a["labels"], b["labels"])
    for k in ("prior_clean", "baseline_clean", "followup_clean"):
        assert np.allclose(a[k], b[k])
    assert not np.allclose(a["baseline"], b["baseline"])
    assert a["dt"] == b["dt"] and a["dt_prior"] == b["dt_prior"]


def test_three_scans_in_time_order():
    from admod.dataset import make_subject
    s = make_subject((7, "typical", (1.0, 4.0), 32, 1.0))
    ctx = (s["labels"] >= 1) & (s["labels"] <= 16)
    m = [s[k][ctx].mean() for k in ("prior_clean", "baseline_clean", "followup_clean")]
    assert m[0] <= m[1] <= m[2]


def test_more_amyloid_means_higher_cortical_suvr(brain):
    tissue = dict(sample_tissue(np.random.default_rng(0)), noise=0.0)
    gm = brain["cortex_gm"]
    lo, _ = scan(brain["labels"], np.zeros(gm.shape), tissue, np.random.default_rng(1), global_sd=0.0, max_shift_vox=0.0)
    hi, _ = scan(brain["labels"], gm * 0.8, tissue, np.random.default_rng(1), global_sd=0.0, max_shift_vox=0.0)
    assert hi[gm].mean() > lo[gm].mean() + 0.5


def test_params_positive():
    p = sample_params(np.random.default_rng(0))
    assert p["rho"] > 0 and p["eps"] > 0 and p["D"] > 0
    assert VOXEL_MM == 5.0 and len(REGIONS) == 8
