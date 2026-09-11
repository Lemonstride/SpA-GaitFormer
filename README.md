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
