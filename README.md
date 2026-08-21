# SpA-GaitFormer

SpA-GaitFormer is the reproducible implementation for the SpA-MMD
paper. It implements the paper's RGB ViT, SkeletonGait++ feature adapter,
mmWave range-Doppler (RD) temporal Transformer, strict frame-feature-level 3:1
alignment, cross-modal Transformer, supervised cross-entropy training, and
subject-independent evaluation.

## Reproduction status

The following parts are implemented and testable:

- RGB frames are encoded independently by the original ViT-B/16 architecture.
- SkeletonGait++ frame features are projected into the shared token dimension.
- Every three RGB and three skeleton frame features are mean-pooled to match one
  RD frame. Shape checks reject any sequence that violates the exact 3:1 ratio.
- RD maps are encoded per frame and then modeled by a temporal Transformer.
- Modality and temporal embeddings plus a cross-modal Transformer produce the
  clip representation and classification logits.
- Binary and four-class tasks use separate models and unweighted batch-mean
  cross-entropy, Adam with learning rate `1e-4`, batch size 16, and 50 epochs.
- Subject-independent repeated splits, macro metrics, checkpoints, ablations,
  and strict window counting are provided.

The uploaded public data currently contains only `S01`, whose labels are
`unknown`. Its raw radar `.bin` files are present, but the acquisition profile
needed to reconstruct RD maps is absent. Therefore this repository can validate
the data path and model path, but it cannot honestly reproduce the paper's
30-subject, 3,595-window metrics until the following experiment assets are added:

1. Clinical labels for all subjects.
2. All subject folders used in the paper.
3. The exact window length and stride used in the reported experiments.
4. Radar acquisition parameters such as ADC samples, chirps, RX/TX layout,
   sample rate, slope, and FFT settings.
5. The pose-landmarker asset and the actual ViT/SkeletonGait++ checkpoints.

No script silently invents these values. Formal commands fail with an actionable
message when required evidence is missing.

## Repository layout

```text
configs/                 formal and smoke-test configurations
scripts/                 preprocessing and SkeletonGait++ extraction helpers
spa_gaitformer/          model, data, training, metrics, and CLI code
tests/                   unit and tensor-interface tests
third_party/OpenGait/    pinned official OpenGait source snapshot
```

## Install

Clone the repository together with its pinned OpenGait dependency:

```bash
git clone --recurse-submodules https://github.com/Lemonstride/SpA-GaitFormer.git
cd SpA-GaitFormer
python -m pip install -e '.[dev,opengait]'
```

For an existing checkout, initialize or refresh the submodule with:

```bash
git pull
git submodule update --init --recursive
```

## Data contract

Each manifest row identifies one synchronized clip. The model expects:

- RGB: `[3T, 3, H, W]`
- SkeletonGait++ frame features: `[3T, D_skeleton]`
- RD maps: `[T, 1, H_rd, W_rd]`

The feature extractor stores skeleton features before multimodal pooling. The
model itself performs both three-frame pooling operations, making the 3:1 rule
explicit and testable rather than relying on filenames or nominal frame rates.

Generate a window manifest only after RD and skeleton features exist:

```bash
spa-build-manifest \
  --processed-root /path/to/SpA-MMD-processed \
  --labels-csv /path/to/clinical_labels.csv \
  --rd-root /path/to/rd_maps \
  --skeleton-root /path/to/skeleton_frame_features \
  --rd-window 30 --rd-stride 15 \
  --output manifests/all.csv
```

The values `30` and `15` above are examples, not recovered paper settings.

Create five subject-independent split sets:

```bash
spa-make-splits --manifest manifests/all.csv --output-dir manifests/splits \
  --repeats 5 --train-ratio 0.70 --val-ratio 0.15
```

The ratios are also explicit inputs because the manuscript does not currently
state the original values.

## Training and evaluation

Fill the `null` experiment parameters and local checkpoint paths in
`configs/spa_mmd.yaml`, then run binary and severity tasks separately:

```bash
spa-train --config configs/spa_mmd.yaml \
  --train-manifest manifests/splits/split_0_train.csv \
  --val-manifest manifests/splits/split_0_val.csv \
  --task binary --output-dir outputs/binary_split_0

spa-evaluate --config configs/spa_mmd.yaml \
  --manifest manifests/splits/split_0_test.csv \
  --task binary --checkpoint outputs/binary_split_0/best.pt
```

For interface validation without paper assets, use `configs/smoke.yaml` and
`scripts/make_synthetic_dataset.py`. Smoke results are software checks only.

## Third-party provenance

`third_party/OpenGait` is a Git submodule pointing to a pinned commit from the
official OpenGait project; provenance is recorded in `THIRD_PARTY.md`.
SkeletonGait++ is used as a convolutional/P3D front-end whose extracted frame
features are projected into SpA-GaitFormer's Transformer token space; it is not
described as a Transformer itself.
