# Dense-Window Head-Turn Experiment

This document records the final 80%-overlap, from-scratch paired experiment
completed on 2026-09-15. It is separate from the archived pre-training and
fine-tuning results and from the earlier 50%-overlap smoke experiment.

## Protocol

- Walking windows contain 30 RGB frames, 30 skeleton frames, and 10 RD maps.
- RGB and skeleton strides are 6 frames and the RD stride is 2 maps, giving
  80% overlap while preserving strict frame-level `3:3:1` correspondence.
- The strict cohort contains 20 participants and 2,020 walking windows. The
  all-recorded sensitivity cohort contains 25 complete cases and 2,437 windows.
- Each cohort uses five repeated subject-independent 70/15/15 holdouts for the
  binary and four-stratum tasks. These are repeated holdouts, not five-fold
  cross-validation.
- Each task is trained both without and with the independent participant-level
  head-turn token, producing 40 runs of 50 epochs.
- No pre-trained weights are loaded. Training uses deterministic 32-view RGB
  and geometry augmentation, participant-and-class-balanced sampling, mixed
  precision, physical batch size 2, and gradient accumulation over 8 steps.
- Checkpoints are selected by validation participant-level macro-F1. Test
  probabilities are averaged over all windows of each participant.

## Participant-Level Results

Values are percentages reported as mean +/- sample standard deviation over the
five repeated holdouts.

| Cohort | Task | Walk accuracy | Walk macro-F1 | +Head-turn accuracy | +Head-turn macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Strict review (`n=20`) | AS versus control | 75.00 +/- 17.68 | 58.48 +/- 27.68 | 90.00 +/- 13.69 | 77.14 +/- 31.30 |
| Strict review (`n=20`) | Four strata | 35.00 +/- 22.36 | 25.33 +/- 23.29 | 50.00 +/- 17.68 | 37.00 +/- 20.36 |
| All recorded (`n=25`) | AS versus control | 66.67 +/- 0.00 | 45.33 +/- 11.93 | 80.00 +/- 18.26 | 64.00 +/- 32.86 |
| All recorded (`n=25`) | Four strata | 35.00 +/- 13.69 | 22.83 +/- 13.91 | 35.00 +/- 22.36 | 24.50 +/- 23.74 |

The mean paired head-turn-minus-walk changes were +15.00 accuracy points and
+18.67 macro-F1 points for strict binary classification, +15.00 and +11.67 for
strict four-stratum classification, +13.33 and +18.67 for all-recorded binary
classification, and 0.00 and +1.67 for all-recorded four-stratum
classification.

## Acceptance and Boundaries

The final audit accepted execution and partition integrity:

- all 40 jobs exited with code 0;
- all 40 participant-level metric files and 40 non-empty checkpoints exist;
- train, validation, and test participants are disjoint in every split;
- prediction identities and per-participant window counts match the test
  manifests;
- paired variants use the same held-out participants;
- head-turn normalization matches unique training-participant statistics; and
- no traceback or CUDA out-of-memory marker was found.

Single-class participant predictions occurred in 15 of the 40 held-out
evaluations. Each test partition contains only three or four participants, and
the standard deviations are correspondingly large. The mean changes are
therefore descriptive. They do not establish a generalisable head-turn benefit
or participant-level clinical performance, and repeating the same protocol
would not correct this independent-sample limitation.

The reproducible audit entry point is
`scripts/audit_dense_headturn_results.py`. The full AML run is stored under the
user-owned experiment directory; checkpoint binaries and source clinical data
are not included in this repository.

