# Reproduction Status

This document records the independently reproduced, walk-only, from-scratch
baseline completed on 2026-09-10. It is separate from the manuscript's archived
pre-training and fine-tuning experiments.

## Cohort and Inputs

- Source cohort: 26 participants, including 19 patients with AS and 7 healthy
  controls.
- Complete-case cohort: 25 participants. One walking radar recording was too
  short to form one complete tri-modal window and was excluded before splitting.
- Strictly aligned walking windows: 973.
- Binary windows: 286 healthy-control and 687 AS.
- Four-class windows: 286 healthy, 229 mild, 336 moderate, and 122 severe.
- Each window contains 30 RGB frames, 30 depth-derived skeleton frames, and 10
  RD maps. The corresponding strides are 15, 15, and 5 frames.
- The `3:3:1` relationship is enforced on frame-level features from the same
  physical interval.

## Model and Training

- RGB encoder: ViT-B/16 with `weights=None`.
- Skeleton encoder: official SkeletonGait++ P3D front end without a checkpoint;
  its 4,096-dimensional output is projected into the shared token space.
- Radar encoder: CNN plus two-layer temporal Transformer.
- Shared representation: 256 dimensions.
- Fusion: three-layer, eight-head cross-modal Transformer with a learned `[CLS]`
  token and modality/time embeddings.
- Training: 50 epochs, mixed precision, physical batch size 2, and gradient
  accumulation over 8 steps on one NVIDIA A100-SXM4 80GB GPU.
- Augmentation is deterministic and training-only. Validation and test samples
  are not augmented.
- Sampling gives equal total mass to each class and equal participant mass
  within each class.

## Evaluation Protocol

Five repeated subject-independent holdouts were used; this is not five-fold
cross-validation. Binary splits contain 18/4/3 train/validation/test
participants, and four-class splits contain 17/4/4. Every window from one
participant remains in one partition.

The best checkpoint is selected by validation participant-level macro-F1. At
test time, softmax probabilities are averaged over all valid windows from each
participant before assigning one participant-level prediction. Reported
dispersion is the sample standard deviation across the five holdouts.

## Participant-Level Results

| Task | Accuracy (%) | Macro-precision (%) | Macro-recall (%) | Macro-F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| AS versus control | 53.33 +/- 18.26 | 30.00 +/- 4.56 | 40.00 +/- 13.69 | 34.00 +/- 8.22 |
| Four functional strata | 20.00 +/- 20.92 | 10.42 +/- 15.59 | 20.00 +/- 20.92 | 12.83 +/- 17.09 |

The four-class accuracies were 50%, 0%, 0%, 25%, and 25%. Late training
performance approached saturation while held-out participant performance
remained low and split-sensitive. These results are a negative sensitivity
baseline: random initialisation did not provide reliable participant-level
generalisation in this small complete-case cohort.

## Verification

All ten runs completed. The final audit passed the following checks for every
run:

- train, validation, and test participant sets are disjoint;
- prediction participants match the test manifest;
- participant and window counts are consistent;
- probabilities are finite and sum to one;
- saved predictions match probability `argmax`;
- participant predictions use mean softmax probability over all valid windows.

The de-identified text artifacts and integrity inventories are published under
`results/from_scratch_26cohort_complete25_walk_v1`. Internal participant IDs are
mapped to stable public codes with a private, non-exported salt. Checkpoints are
not published; `CHECKPOINTS.sha256` records their hashes and sizes.

## Boundaries

- These participant-level results must not be combined numerically with the
  archived pre-trained window-level results.
- The experiment is walk-only and contains no head-turn feature.
- Increasing window overlap would not increase the number of independent
  participants.
- Radar physical range and velocity axes require the original acquisition
  profile. A provisional FFT layout is not a confirmed calibration.
- This software and its outputs are research artifacts, not a clinical
  diagnostic system.
