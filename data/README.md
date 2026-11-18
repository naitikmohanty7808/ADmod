# Data

All data in this study are synthetic. The cohort files (about 3.9 GB) are not stored here; they are regenerated
exactly from the generator seeds in `protocol/protocol.json`:

```bash
python -m admod.dataset            # training, validation, development, sweep and sensitivity cohorts
python -m admod.dataset --confirm  # confirmatory cohorts C1-C3 (seeds 21-23)
```

`protocol/CONFIRM_COHORT_DIGESTS.txt` lists the SHA-256 content digests of the confirmatory cohorts;
`python -m admod.posthoc` recomputes and checks them. No OASIS, ADNI or other human data are used.
