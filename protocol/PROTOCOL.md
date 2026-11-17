# Confirmatory protocol (frozen)

All data are synthetic. `protocol.json` is the machine-readable version; this file explains it.

## Why a frozen protocol

ADmod and the baselines were developed while looking at the medRxiv v2 test sets
(`test`, `stress_atypical`, `stress_long_gap`, `stress_long_gap_fresh`). Results on those sets are
exploratory. Confirmatory claims in the IEEE Access paper come only from three new cohorts that did
not exist when this protocol was committed, evaluated by an analysis fixed here.

Order of events, each recorded as a git commit:

1. This protocol and the training code are committed. No ablation model has been trained.
2. All 21 models (7 variants x 3 seeds) and the 12 sensitivity models are trained under one policy.
3. The confirmatory analysis code (`admod/confirm.py`) is committed.
4. Only then are the confirmatory cohorts (generator seeds 21, 22, 23) generated and scored, once.

## Confirmatory cohorts

| ID | File | n | Seed | Spread pattern | Forecast gap |
|----|------|---|------|----------------|--------------|
| C1 | confirm_id.npz | 400 | 21 | typical (training pattern) | 1-4 y |
| C2 | confirm_atypical.npz | 200 | 22 | atypical (never trained on) | 1-4 y |
| C3 | confirm_long.npz | 200 | 23 | typical | 6-8 y |

## Models

Seven variants, seeds 0, 1, 2, all trained 100 epochs with AdamW (lr 2e-3, weight decay 1e-4),
batch 4, cosine annealing to zero, left-right mirroring, lowest-validation-loss checkpoint,
bfloat16 autocast during training and float32 for every evaluation.

- `ax_full`: ADmod.
- `ax_plain`: plain 3D U-Net with the same backbone and gap channels; no global context, no region
  maps, no coordinates, one direct correction head.
- `ax_no_gap`, `ax_no_global`, `ax_no_region`, `ax_one_scan`, `ax_direct`: ADmod with one component
  removed.

Baselines are fit on the training set only. The ridge penalty is chosen on the validation set from
an extended grid (1e-4 to 1e3).

## Primary endpoint and hypotheses

Endpoint: regional change R^2 against the measured follow-up (16 bilateral cortical regions,
pooled over subjects). Statistic: Delta = R^2(ADmod) - R^2(comparator), deep-model R^2 averaged
over the 3 seeds. Inference: 10,000 paired subject-bootstrap resamples; one-sided
p = (1 + #{Delta* <= delta0}) / (B + 1). Holm step-down over four tests, one-sided alpha = 0.025.

| | Cohort | Comparator | Test | delta0 |
|--|--|--|--|--|
| P1 | C1 | Regional linear + WM | superiority | 0 |
| P2 | C1 | Plain 3D U-Net | superiority | 0 |
| P3 | C1 | Cross-region ridge | non-inferiority | -0.05 |
| P4 | C2 | Cross-region ridge | superiority | 0 |

Why these four: P1 tests ADmod against the best regional model; P2 tests whether ADmod's design
beats a generic conditioned 3D U-Net; P3 and P4 test the trade-off seen in exploratory data, where
the cross-region ridge matched ADmod in distribution and failed under a changed spread pattern.

## Ablation family

C1, full ADmod against each of the six other variants, two-sided bootstrap p, Holm over six,
alpha = 0.05.

## Secondary and sensitivity analyses (descriptive)

See `protocol.json`. These carry no confirmatory claims.

## Deviations

Any change after freezing goes in `DEVIATIONS.md` and in the paper.
