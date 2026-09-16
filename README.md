# SpA-GaitFormer

SpA-GaitFormer is the research implementation accompanying the SpA-MMD
multimodal movement-assessment study. It combines RGB video, depth-derived
skeleton representations, and 60 GHz millimeter-wave range--Doppler (RD) maps
for ankylosing-spondylitis (AS) versus healthy-control classification and
four-class functional-stratum classification.

This repository is provided for inspection and research verification under the
terms in [LICENSE.md](LICENSE.md). It is not a clinical diagnostic system.

## License Attention

The source is visible for inspection, academic reference, and citation only.
No permission is granted to run, reuse, modify, redistribute, or incorporate
this code into another project except where separately required by an
identified third-party license. See [LICENSE.md](LICENSE.md) for the controlling
terms.

## Architecture

- Original ViT-B/16 encodes each `224 x 224` RGB frame into a 768-dimensional
  representation.
- The official SkeletonGait++ P3D front end encodes pose-heatmap and silhouette
  inputs and exposes 4,096-dimensional frame features.
- A CNN followed by a two-layer temporal Transformer encodes RD maps.
- Linear projection and normalisation map all branches to a shared
  256-dimensional space.
- Three RGB features, three skeleton features, and one RD feature from the same
  physical interval are aligned with a strict frame-level `3:3:1` rule.
- A three-layer, eight-head cross-modal Transformer with modality/time
  embeddings and a learned `[CLS]` token produces the fused representation.
- Binary and four-class models are trained separately with unweighted
  cross-entropy.

SkeletonGait++ is used as a convolutional/P3D front end. Its output is projected
into the shared token space before cross-modal Transformer fusion; it is not
described as a Transformer architecture itself.

## Verified From-Scratch Baseline

The repository includes an independent walk-only sensitivity experiment that
is separate from the paper's archived pre-training and fine-tuning results.

- Source cohort: 26 participants, including 19 patients with AS and 7 healthy
  controls.
- Complete-case cohort: 25 participants. One walking radar recording was too
  short to form a complete tri-modal window and was excluded before splitting.
- Data: 973 strictly aligned walking windows, comprising 286 healthy-control
  and 687 AS windows. Four-class counts are 286 healthy, 229 mild, 336 moderate,
  and 122 severe windows.
- Window structure: 30 RGB frames, 30 skeleton frames, and 10 RD maps; strides
  are 15, 15, and 5 frames, respectively.
- Initialisation: no transferred ViT or SkeletonGait++ weights; all model
  components are trained from random initialisation.
- Evaluation: five repeated subject-independent holdouts. Binary splits contain
  18/4/3 train/validation/test participants; four-class splits contain 17/4/4.
- Selection and aggregation: the best checkpoint is selected by validation
  subject-level macro-F1. Test-window softmax probabilities are averaged per
  participant before assigning one participant-level prediction.

The aggregate metrics, provenance, integrity audit, and checkpoint hashes are
browsable under
[`results/from_scratch_26cohort_complete25_walk_v1`](results/from_scratch_26cohort_complete25_walk_v1).
The complete de-identified bundle of manifests, histories, run configurations,
test predictions, logs, and audits is provided as
[`results/from_scratch_26cohort_complete25_walk_v1.tar.gz`](results/from_scratch_26cohort_complete25_walk_v1.tar.gz).
These results quantify a small-cohort, randomly initialised baseline and must
not be compared directly with experiments that use different tasks, cohorts,
pre-training, or evaluation units.

## Preliminary Reviewed Head-Turn Experiment

A preliminary paired follow-up compares walk-only training with the same model
augmented by one reviewed head-turn angular-range token. It uses 50% window
overlap and is retained as smoke evidence rather than a final manuscript
result. The strict `3:3:1` walk stream is unchanged. The head-turn value is
normalised using training participants only and enters the fusion Transformer
as an independent clinical kinematic token.

The eligible subset contains 20 participants and 806 walk windows. Across five
repeated subject-independent holdouts, adding the token changed binary
participant-level accuracy from 60.00% to 80.00% and macro-F1 from 42.48% to
63.81%. Four-class accuracy changed from 30.00% to 50.00% and macro-F1 from
22.00% to 37.83%. These are descriptive means from four-participant test sets,
not confirmatory estimates.

The full protocol, dispersion, failure modes, and interpretation boundaries are
documented in [`docs/HEADTURN_EXPERIMENT.md`](docs/HEADTURN_EXPERIMENT.md). The
de-identified text artifacts are under
[`results/from_scratch_walk_headturn_primary20_v1`](results/from_scratch_walk_headturn_primary20_v1),
with the complete archive at
[`results/from_scratch_walk_headturn_primary20_v1.tar.gz`](results/from_scratch_walk_headturn_primary20_v1.tar.gz).

## Final Dense-Window Head-Turn Experiment

The final paired matrix uses 80% window overlap and contains 40 from-scratch
runs across strict-review and all-recorded cohorts, binary and four-stratum
tasks, walk-only and walk-plus-head-turn variants, and five repeated
subject-independent holdouts. All runs, metrics, checkpoints, partition checks,
and training-only head-turn normalization checks passed the integrity audit.

In the strict 20-participant cohort, head-turn fusion changed mean binary
participant-level macro-F1 from 58.48% to 77.14% and four-stratum macro-F1 from
25.33% to 37.00%. In the 25-participant sensitivity cohort, the corresponding
changes were 45.33% to 64.00% and 22.83% to 24.50%. Fifteen of 40 held-out
evaluations predicted only one class across their three or four test
participants, so these values are descriptive and do not establish clinical
efficacy.

See [`docs/DENSE_HEADTURN_EXPERIMENT.md`](docs/DENSE_HEADTURN_EXPERIMENT.md),
[`results/from_scratch_walk_headturn_dense_8gpu_v2`](results/from_scratch_walk_headturn_dense_8gpu_v2),
and `scripts/audit_dense_headturn_results.py` for the final protocol, concise
results, and reproducible acceptance checks.

## Training Augmentation

The verified baseline uses the deterministic `subject_cycle_rgb32_v1` policy on
training samples only. Each participant cycles through 32 label-blind views over
epochs. A single view is applied consistently to every frame in a window:

- RGB uses mild brightness, contrast, saturation, and gamma changes together
  with optional horizontal mirroring and non-cropping shrink/pad geometry.
- Pose heatmaps and silhouettes receive the corresponding mirror and geometry
  transformation.
- RD maps receive per-window normalisation but no random noise or amplitude
  perturbation in this experiment.
- Temporal reversal, dropping, and masking are disabled.

Validation and test windows are not augmented. The training sampler assigns
equal total mass to each class and equal participant mass within each class.

## Repository Layout

```text
configs/                 experiment and smoke-test configurations
docs/                    verification notes and methodological boundaries
results/                 de-identified experiment artifacts
scripts/                 preprocessing, execution, and export helpers
spa_gaitformer/          data, model, training, evaluation, and metrics code
tests/                   unit and tensor-interface tests
third_party/OpenGait/    pinned OpenGait Git submodule
```

## Installation

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,opengait]"
```

Use an environment with a CUDA-enabled PyTorch build for GPU training. The code
also supports CPU smoke tests, although the full model is computationally heavy.

## Data Contract

Each manifest row identifies one synchronized window and contains paths to RGB
frames, skeleton tensors, and RD tensors plus temporal indices and task labels.
The model expects:

- RGB: `[3T, 3, H, W]`
- Skeleton input: `[3T, 3, 64, 44]` for the SkeletonGait++ backend, or cached
  frame features `[3T, 4096]`
- RD maps: `[T, 1, H_rd, W_rd]`

Shape checks reject incomplete windows or sequences that violate the exact
`3:3:1` relationship.

Generate a manifest after all modality features are available:

```bash
spa-build-manifest \
  --processed-root /path/to/processed-rgb \
  --labels-csv /path/to/clinical_labels.csv \
  --rd-root /path/to/rd-maps \
  --skeleton-root /path/to/skeleton-features \
  --rd-window 10 \
  --rd-stride 5 \
  --sessions walk \
  --output manifests/all_walk.csv
```

Create subject-independent splits and run one task:

```bash
spa-make-splits \
  --manifest manifests/all_walk.csv \
  --output-dir manifests/binary \
  --task binary \
  --repeats 5 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --seed 2026

spa-train \
  --config configs/from_scratch_26_walk.yaml \
  --train-manifest manifests/binary/split_0_train.csv \
  --val-manifest manifests/binary/split_0_val.csv \
  --task binary \
  --output-dir outputs/binary_split_0

spa-evaluate \
  --config configs/from_scratch_26_walk.yaml \
  --manifest manifests/binary/split_0_test.csv \
  --task binary \
  --checkpoint outputs/binary_split_0/best.pt \
  --output outputs/binary_split_0/test_metrics.json
```

The batch runners accept `SPA_GAITFORMER_ROOT`, `PYTHON_BIN`, `RUN_DIR`,
`CONFIG_PATH`, `CUDA_DEVICE`, and `DEVICE` environment variables and do not rely
on a particular computing platform.

For the denser-window sensitivity protocol, use
`configs/from_scratch_26_walk_dense.yaml` for the walk-only comparator and
`configs/from_scratch_26_walk_headturn_dense.yaml` for the head-turn variant.
Both keep a 10-map RD window but reduce RD stride from 5 to 2; the corresponding
RGB and skeleton stride changes from 15 to 6 while preserving exact frame-level
`3:3:1` alignment. Dense windows must still be split by participant, never by
window.

On a dedicated eight-GPU host, the complete dense matrix can be launched with:

```bash
export SPA_USER_ROOT=/path/to/writable/root
bash scripts/start_headturn_dense_8gpu.sh
```

The matrix runs the 20-participant strictly reviewed cohort as the primary
analysis and the 25-participant all-recorded cohort as a sensitivity analysis.
For each cohort it runs both tasks, five subject-independent holdouts, and both
the walk-only and walk-plus-head-turn variants, for 40 independent 50-epoch
jobs in total. One job is assigned to each GPU at a time. Completed
`test_metrics.json` files are validated and skipped when the same run directory
is resumed.

## Publishing Result Artifacts

`scripts/prepare_public_results.py` creates a portable text-only archive. It
replaces private absolute roots, maps internal participant IDs to stable public
codes using a private non-exported salt, records checkpoint hashes, and produces
an `ARTIFACTS.sha256` inventory. Model checkpoints are deliberately excluded
from the GitHub archive because each full-model checkpoint is approximately
397 MB.

```bash
python scripts/prepare_public_results.py \
  --run-dir /path/to/private-run \
  --output-dir /path/to/public-export \
  --config configs/from_scratch_26_walk.yaml \
  --redact-root /path/to/private-root
```

Synthetic smoke results validate software interfaces only and are never
reported as research performance. Radar-derived physical axes also require the
archived acquisition profile; provisional FFT layouts must not be presented as
confirmed distance or velocity calibration.

## Third-Party Provenance

`third_party/OpenGait` points to the official OpenGait source at the commit
recorded in [THIRD_PARTY.md](THIRD_PARTY.md). OpenGait remains subject to its own
license and upstream terms.

