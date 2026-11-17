# ADmod: amyloid-PET forecasting under simulated measurement error and distribution shift

Code, protocol, trained models and results for the IEEE Access article
"ADmod: A Temporally Conditioned 3D Encoder-Decoder for Amyloid-PET Forecasting Under Simulated
Measurement Error and Distribution Shift" (N. Mohanty, M. Sarmadi).

**All data are synthetic.** No OASIS, ADNI or other human data are read, copied or distributed.
Results describe what a forecaster can learn under the simulator's stated assumptions. They make
no claim about patients.

## Exploratory versus confirmatory

| Status | What | Where |
|---|---|---|
| Confirmatory | Primary hypotheses P1 to P4 and the ablation family on fresh cohorts C1 to C3 | `protocol/`, `admod/confirm.py`, `results/confirm.json` |
| Descriptive (in protocol) | 12 simulator sensitivity settings with retraining | `results/sensitivity.json` |
| Post hoc (added before scoring) | digest and checkpoint checks, bias-corrected reference, P2 seed pairings, gap-bin RMSE, timing, bfloat16 parity run | `admod/posthoc.py`, `results/posthoc.json`, `runs_parity/` |
| Exploratory | Previous revision's development cohorts (S2) and the reanalysis of the earlier classification study (S1) | `admod/explore_s2.py`, `admod/eaddls.py` |

The protocol, hypotheses, statistics and training code were committed before training
(`protocol/FREEZE_COMMIT.txt`), the analysis code next (`protocol/ANALYSIS_COMMIT.txt`), then the
content digests of the confirmatory cohorts (`protocol/COHORT_COMMIT.txt`). Every departure from the
protocol, including the loss and restoration of the original repository, is listed in
`protocol/DEVIATIONS.md`.

## Install

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.15.0 (CPU), NumPy 2.5, SciPy 1.18, scikit-learn 1.9, Matplotlib 3.11,
on 2 cores of an Intel Xeon (2.3 GHz) with AMX (bfloat16) units and 8 GB RAM. No GPU is needed.

## Reproduce

```bash
python -m pytest -q                       # unit tests
python -m admod.dataset                   # training and exploratory cohorts (generator seeds in protocol.json)
./run_ax.sh                               # sensitivity cohorts, 21 + 12 training runs, parity runs (resumable)
python -m admod.dataset --confirm         # confirmatory cohorts C1-C3 (seeds 21-23)
python -m admod.posthoc                   # digest and checkpoint checks
python -m admod.confirm                   # confirmatory analysis -> results/confirm.json
python -m admod.confirm --sensitivity     # sensitivity analysis -> results/sensitivity.json
python -m admod.posthoc --analyze         # post hoc analyses -> results/posthoc.json
python -m admod.calibrate                 # simulator calibration -> results/calibration.json
python -m admod.eaddls                    # S1 -> results/eaddls.json
python -m admod.explore_s2                # S2 -> results/explore_s2.json and supplement tables
python -m admod.make_tex_ax --out ../paper_access        # every number, table and figure of the article
python -m admod.eaddls_tex --out ../paper_access/supp    # S1 numbers and tables
```

Each training run writes `runs_ax/<variant>_s<seed>/best.pt` (lowest validation loss) and
`train_log.json` (per-epoch losses and wall times). Training uses bfloat16 autocast; every evaluation
runs in float32. Runtimes are reported in the article and in `results/posthoc.json`.

On a 2-core machine, do not run another CPU-heavy job during training: thread contention slows
training about tenfold. `side_run.sh <command>` pauses training, runs the command and resumes.

## Layout

| Path | Role |
|---|---|
| `admod/phantom.py` | 32^3 brain phantom, 16 cortical regions, per-subject anatomy |
| `admod/amyloid.py` | Fisher-KPP accumulation and spread; typical and atypical spread patterns |
| `admod/pet.py` | PET formation: blur, noise, global gain, misregistration, cerebellar-GM SUVR |
| `admod/dataset.py` | Cohort generation, content digests (`array_digest`) |
| `admod/models.py` | ADmod and its six ablation variants |
| `admod/baselines.py` | Copy last scan, regional models, cross-region ridge |
| `admod/train.py` | Training policy shared by all variants |
| `admod/confirm.py`, `admod/posthoc.py` | Confirmatory, sensitivity and post hoc analyses |
| `admod/calibrate.py` | Scan-rescan variability and accumulation rates against published values |
| `admod/eaddls.py`, `admod/explore_s2.py` | Supplementary Sections S1 and S2 |
| `admod/make_tex_ax.py`, `admod/eaddls_tex.py` | LaTeX macros, tables and figures from result files only |
| `protocol/` | Frozen protocol, commit records, cohort digests, deviations |

## License

Code: MIT License (see `LICENSE`). Synthetic data, trained models and results: CC BY 4.0.
