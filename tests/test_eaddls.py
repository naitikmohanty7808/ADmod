"""S1 cohort: design counts, CDR as label source, determinism."""
import numpy as np

from admod.eaddls import check_design, first_visit, make_cohort, task_labels


def test_design_counts_and_cdr_label_source():
    c = make_cohort(np.random.default_rng(4100))
    info = check_design(c)  # asserts that the CDR sequence defines the groups
    assert info["groups"] == [72, 64, 14]
    c1 = first_visit(c)
    _, y = task_labels(c1, "detection")
    assert np.array_equal((c1["cdr"] > 0).astype(int), y)  # CDR > 0 at the first visit is the detection label


def test_cohort_is_deterministic():
    a, b = make_cohort(np.random.default_rng(7)), make_cohort(np.random.default_rng(7))
    assert all(np.array_equal(a[k], b[k]) for k in a)
