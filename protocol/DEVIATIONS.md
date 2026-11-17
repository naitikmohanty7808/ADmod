# Deviations from the frozen protocol

## Loss of the original repository and restoration

- **2026-11-16, 21:50 PST.** The protocol and training code were first committed (original repository,
  commit `ed118d1`). The analysis code followed at 21:55 (`2629482`), the confirmatory cohorts were
  generated and their file hashes committed at about 22:00 (`98b8dcb`), and the post hoc analysis code
  at 23:10 (`52fdb11`).
- **2026-11-17, 00:21 PST.** The cloud compute environment was replaced and every file in it was lost:
  the repository, all generated data, and the models trained so far. No confirmatory cohort had been
  scored. Five seed-0 models had finished training. The validation results of four of them (ADmod, plain
  3D U-Net, without gap input, without global context) had been inspected: on the validation set,
  the plain 3D U-Net and the variant without gap input reached lower validation loss than ADmod.
  Nothing in the protocol was changed as a result.
- **Restoration.** The simulator, cohort generator and baselines were restored from the record of the
  earlier work session and verified: the regenerated training cohort is identical to the original
  (bit-for-bit check of a stored subject was not possible, so instead every classical baseline's
  regional change R^2 on the regenerated development test set and the ridge validation score
  (0.39097500973797683) match the previous version's results to all reported digits). The revision
  code (ablation flags, training policy, cohort functions, analysis code, post hoc code) and the
  protocol files were restored with identical content, recommitted in a new repository, and all models
  are retrained from scratch under the unchanged protocol. Because the original commits are lost, the
  restored commits postdate the loss; their content matches the originals.
- PyTorch changed from 2.14.1 to 2.15.0 because the environment was rebuilt.
- File hashes of `.npz` cohorts cannot be reproduced (numpy writes zip timestamps), so the
  confirmatory cohorts are now fingerprinted by content digests (`admod.dataset.array_digest`).

## Deviations recorded before the loss (still apply)

1. **Bug fix in `admod/dataset.py` (no effect on any number).** The `if __name__ == "__main__"` guard
   originally sat above the new cohort functions; it is at the end of the file.
2. **Order of events.** PROTOCOL.md lists the analysis commit and cohort generation after model
   training. Both happen before training finishes. Training reads only `data/train.npz` and
   `data/val.npz`; no script loads the confirmatory cohorts between generation and the single scoring
   run other than to compute their content digests.

# Post hoc additions (not part of the frozen protocol)

Added after an internal review of the methods, before any confirmatory scoring (`admod/posthoc.py`):
a content-digest check of the confirmatory cohorts and a checkpoint check before scoring; a
bias-corrected noise-free reference; the nine seed pairings for P2; regional RMSE per forecast-gap
bin; median epoch times; and a same-session bfloat16 versus float32 parity run. All are descriptive.
