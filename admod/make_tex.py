"""LaTeX macros for the simulator calibration (restored for the IEEE Access revision)."""
from __future__ import annotations


def calib_macros(cal: dict) -> list[str]:
    out = []

    def m(name, val):
        out.append(rf"\newcommand{{\{name}}}{{{val}}}")
    t = cal["test_retest"][0]
    m("calComp", f"{t['composite_mean_abs_pct_diff']:.1f}")
    m("calCompSD", f"{t['composite_sd']:.1f}")
    m("calReg", f"{t['regional_mean_abs_pct_diff']:.1f}")
    m("calN", t["n_subjects"])
    acc = cal["accumulation"]
    m("accN", acc["n_subjects"])
    bins = acc["bins"]
    lo, hi = bins[0], bins[-1]
    pk = max(bins, key=lambda b: b["rate"])
    m("accLowRate", f"{lo['rate']:.3f}"); m("accLowHi", f"{lo['hi']:.1f}"); m("accLowN", lo["n"])
    m("accPeakRate", f"{pk['rate']:.3f}"); m("accPeakLo", f"{pk['lo']:.1f}"); m("accPeakHi", f"{pk['hi']:.1f}"); m("accPeakN", pk["n"])
    m("accHighRate", f"{hi['rate']:.3f}"); m("accHighLo", f"{hi['lo']:.1f}"); m("accHighN", hi["n"])
    return out
