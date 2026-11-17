"""Confirmatory analysis helpers: exact metrics and multiplicity correction."""
import numpy as np

from admod.confirm import boot_weights, holm, metric_rows, subject_stats
from admod.dataset import generate_cohort
from admod.metrics import per_subject, scores


def test_fast_metrics_match_reference():
    co = generate_cohort("unit", 4, base_seed=97, workers=1)
    pred = co["baseline"] + 0.5 * (co["followup_clean"] - co["baseline_clean"])
    ref = scores(per_subject(pred, co["followup"], co["baseline"], co["labels"]))
    mine = metric_rows(subject_stats(pred, co), np.ones((1, 4)))
    for k in ("regional_r2", "voxel_r2", "composite_mae"):
        assert abs(ref[k] - mine[k][0]) < 1e-6, k


def test_bootstrap_weights_resample_n_subjects():
    W = boot_weights(7, 50)
    assert W.shape == (50, 7) and np.all(W.sum(1) == 7)


def test_holm():
    out = holm({"a": 0.01, "b": 0.02, "c": 0.04}, alpha=0.05)
    assert np.isclose(out["a"]["p_holm"], 0.03) and np.isclose(out["b"]["p_holm"], 0.04)
    assert np.isclose(out["c"]["p_holm"], 0.04) and all(v["reject"] for v in out.values())
