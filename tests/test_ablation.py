"""Ablation variants (IEEE Access revision)."""
import torch

from admod.models import ADmod, feature_index
from admod.train import AX_VARIANTS, VARIANTS


def _feat(n_scans, B=2):
    torch.manual_seed(0)
    return torch.rand(B, 8 if n_scans == 2 else 5, 32, 32, 32)


def test_every_variant_starts_as_copy_last_scan():
    for v in AX_VARIANTS:
        cfg = VARIANTS[v]
        m = ADmod(cfg["n_scans"], arch=cfg["arch"]).eval()
        f = _feat(cfg["n_scans"])
        out, _ = m(f, torch.tensor([1.0, 3.0]), torch.zeros(2, 2, 32, 32, 32))
        assert torch.equal(out[:, 0], f[:, 0]), v


def test_ablations_remove_inputs():
    assert feature_index(2, True, True) == list(range(8))
    assert feature_index(2, False, True) == [0, 1, 2, 3, 6, 7]
    assert feature_index(2, True, False) == [0, 1, 2, 3, 4, 5]
    assert feature_index(1, True, True) == [0, 1, 2, 3, 4]


def test_no_gap_model_ignores_gap_channels():
    cfg = VARIANTS["ax_no_gap"]
    m = ADmod(2, arch=cfg["arch"]).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    f = _feat(2)
    f2 = f.clone()
    f2[:, 4:6] += 0.5  # change only the gap channels
    dt = torch.tensor([2.0, 2.0])
    pr = torch.zeros(2, 2, 32, 32, 32)
    with torch.no_grad():
        assert torch.allclose(m(f, dt, pr)[0], m(f2, dt, pr)[0])


def test_direct_head_has_one_output_layer():
    m = ADmod(2, arch=VARIANTS["ax_direct"]["arch"])
    assert not hasattr(m, "head_rate") and hasattr(m, "head_fix")


def test_default_architecture_unchanged():
    from admod.models import count_params
    assert count_params(ADmod(2)) == 215402
