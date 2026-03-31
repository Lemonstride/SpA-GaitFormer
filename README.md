# SpA-GaitFormer

Transformer-based dual-modal training pipeline for SpA-MMD, targeting:

- `healthy` / `SpA` binary classification
- disease severity prediction
- `RGB + skeleton(kpt3d)` multimodal fusion

This project was authored on macOS, but the implementation relies on Python, PyTorch, and `pathlib`, so it is intended to run on both Windows and Linux.

## English

### Recommended Small-Sample Workflow

This repository is now designed for small medical datasets with subject-level splits:

1. Keep a fixed test set.
2. Use the training pool for `2-fold` cross-validation.
3. Select a final epoch from cross-validation.
4. Retrain on the full training pool.
5. Evaluate once on the fixed test set.

The default split files already follow this design:

- Training pool: 20 subjects
- Fixed test set: 6 subjects

### Data Assumptions

The current implementation directly targets the SpA-MMD `processed/subject/session` structure and can train on one or more sessions while keeping subject-level separation:

```text
SpA-MMD/
└── processed/
    ├── S01/
    │   ├── walk/
    │   │   ├── rgb/
    │   │   ├── labels/
    │   │   └── skeleton/
    │   │       └── kpt3d/
    │   │           └── kpt3d.npy
    │   └── head_turn/
    │       ├── rgb/
    │       ├── labels/
    │       │   ├── binary_label.txt
    │       │   ├── severity_label.txt
    │       │   ├── disease_annotations.json
    │       │   └── head_turn_state/
    │       │       └── summary.json
    │       └── skeleton/
    │           └── kpt3d/
    │               └── kpt3d.npy
    └── ...
```

Default inputs:

- Sessions: `walk` and `head_turn`
- RGB: `session/rgb/*.png`
- Skeleton: `session/skeleton/kpt3d/kpt3d.npy`
- Default skeleton shape: `T x 33 x 4`, using the first 3 coordinates for training
- Binary label: `session/labels/binary_label.txt`
- Severity label: `session/labels/severity_label.txt`
- Label fallback: `session/labels/disease_annotations.json`

Cross-validation is grouped by subject, so `walk` and `head_turn` from the same subject always stay in the same fold.

### Modalities

Set `model.input_mode` in [configs/spa_mmd_dual_stream.json](/Users/ice/Documents/SpA-GaitFormer/configs/spa_mmd_dual_stream.json) to:

- `fusion`
- `rgb`
- `skeleton`

This lets you run three baselines under the same subject split.

### Project Structure

- [main.py](/Users/ice/Documents/SpA-GaitFormer/main.py): entry point
- [spa_gaitformer/data.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/data.py): subject split loading, fold generation, RGB/skeleton loading
- [spa_gaitformer/model.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/model.py): dual-stream Transformer with modality ablation
- [spa_gaitformer/losses.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/losses.py): multitask loss
- [spa_gaitformer/metrics.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/metrics.py): binary metrics and severity metrics
- [spa_gaitformer/train.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/train.py): cross-validation, final training, and evaluation

### Configuration

Important fields:

- `data.train_split`: subject pool used for cross-validation and final training
- `data.test_split`: held-out test subjects
- `data.sessions`: which sessions to load, for example `["walk", "head_turn"]`
- `model.input_mode`: `fusion`, `rgb`, or `skeleton`
- `model.model_type`: `transformer` or `cnn`
- `cv.num_folds`: default `2`
- `cv.selection_metric`: default `loss`
- `train.balance_disease`: enable balanced sampling for the binary task

### Installation

Windows:

```powershell
python -m pip install -r requirements.txt
```

Linux:

```bash
python -m pip install -r requirements.txt
```

### Commands

Run `2-fold` cross-validation:

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode cross_validate
```

Train the final model on the full training pool:

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode train_final
```

Evaluate on the fixed test set:

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode evaluate --split test
```

### Feature Baseline (Recommended for Small Samples)

Extract hand-crafted features from skeleton + head-turn summary:

```bash
python scripts/extract_spa_mmd_features.py \
  --dataset_root F:/SpA-MMD/processed \
  --sessions walk,head_turn \
  --output_csv D:/projects/SpA-GaitFormer/baselines/spa_mmd_features.csv
```

Train a baseline classifier (binary disease task):

```bash
python scripts/train_baseline_classifier.py \
  --features_csv D:/projects/SpA-GaitFormer/baselines/spa_mmd_features.csv \
  --train_split F:/SpA-MMD/splits/train_subjects.txt \
  --test_split F:/SpA-MMD/splits/test_subjects.txt \
  --task disease \
  --model logreg
```

For severity prediction:

```bash
python scripts/train_baseline_classifier.py \
  --features_csv D:/projects/SpA-GaitFormer/baselines/spa_mmd_features.csv \
  --train_split F:/SpA-MMD/splits/train_subjects.txt \
  --test_split F:/SpA-MMD/splits/test_subjects.txt \
  --task severity \
  --model svm
```

### Windowed Training (Optional)

Generate window manifests (no leakage, subject-level):

```bash
python scripts/build_window_manifest.py \
  --processed_root F:/SpA-MMD/processed \
  --split_file F:/SpA-MMD/splits/train_subjects.txt \
  --sessions walk,head_turn \
  --window 32 \
  --stride 16 \
  --output_csv D:/projects/SpA-GaitFormer/windows/train_windows.csv
```

Then set `data.train_window_manifest` (and optionally test/val) in the config:

```json
"train_window_manifest": "D:/projects/SpA-GaitFormer/windows/train_windows.csv"
```

When window manifests are provided, the loader will slice RGB and skeleton sequences
by `start_frame/end_frame` before sampling to `data.num_frames`.

### Outputs

Cross-validation outputs are written under:

```text
checkpoints/spa_mmd_dual_stream/cv/
```

Final training outputs are written under:

```text
checkpoints/spa_mmd_dual_stream/final/
```

### Suggested Reporting

For binary disease recognition:

- accuracy
- precision
- recall
- F1

For severity prediction:

- severity accuracy
- severity macro-F1

## 中文

### 推荐的小样本流程

当前仓库已经改成适合小样本医学数据的受试者级实验流程：

1. 固定测试集
2. 在训练池内部做 `2-fold` 交叉验证
3. 根据交叉验证选择最终训练轮数
4. 用全部训练池重训最终模型
5. 只在固定测试集上评估一次

默认 split 已经按这个思路写好：

- 训练池：20 人
- 固定测试集：6 人

### 数据假设

当前实现直接适配 SpA-MMD 的 `processed/subject/session` 结构，并支持在保持受试者隔离的前提下同时训练一个或多个 session：

```text
SpA-MMD/
└── processed/
    ├── S01/
    │   ├── walk/
    │   │   ├── rgb/
    │   │   ├── labels/
    │   │   └── skeleton/
    │   │       └── kpt3d/
    │   │           └── kpt3d.npy
    │   └── head_turn/
    │       ├── rgb/
    │       ├── labels/
    │       │   ├── binary_label.txt
    │       │   ├── severity_label.txt
    │       │   ├── disease_annotations.json
    │       │   └── head_turn_state/
    │       │       └── summary.json
    │       └── skeleton/
    │           └── kpt3d/
    │               └── kpt3d.npy
    └── ...
```

默认读取：

- session：`walk` 和 `head_turn`
- RGB：`session/rgb/*.png`
- Skeleton：`session/skeleton/kpt3d/kpt3d.npy`
- 默认 skeleton 形状：`T x 33 x 4`，训练时使用前 3 个坐标维度
- 二分类标签：`session/labels/binary_label.txt`
- 严重程度标签：`session/labels/severity_label.txt`
- 标签回退：`session/labels/disease_annotations.json`

交叉验证会按受试者分组，所以同一个人的 `walk` 和 `head_turn` 一定会落在同一个 fold 里，不会数据泄漏。

### 模态设置

在 [configs/spa_mmd_dual_stream.json](/Users/ice/Documents/SpA-GaitFormer/configs/spa_mmd_dual_stream.json) 里修改 `model.input_mode`，可选：

- `fusion`
- `rgb`
- `skeleton`

这样就可以在同一套受试者划分下做三组 baseline。

### 项目结构

- [main.py](/Users/ice/Documents/SpA-GaitFormer/main.py)：入口脚本
- [spa_gaitformer/data.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/data.py)：受试者 split、fold 划分、RGB/skeleton 读取
- [spa_gaitformer/model.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/model.py)：支持模态消融的 Transformer
- [spa_gaitformer/losses.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/losses.py)：多任务损失
- [spa_gaitformer/metrics.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/metrics.py)：二分类和严重程度指标
- [spa_gaitformer/train.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/train.py)：交叉验证、最终训练、测试评估

### 配置

关键字段：

- `data.train_split`：用于交叉验证和最终训练的受试者池
- `data.test_split`：固定测试集
- `data.sessions`：要读取的 session，例如 `["walk", "head_turn"]`
- `model.input_mode`：`fusion`、`rgb` 或 `skeleton`
- `model.model_type`：`transformer` 或 `cnn`
- `cv.num_folds`：默认 `2`
- `cv.selection_metric`：默认 `loss`
- `train.balance_disease`：二分类任务的平衡采样

### 安装

Windows:

```powershell
python -m pip install -r requirements.txt
```

Linux:

```bash
python -m pip install -r requirements.txt
```

### 运行命令

运行 `2-fold` 交叉验证：

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode cross_validate
```

在全部训练池上训练最终模型：

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode train_final
```

在固定测试集上评估：

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode evaluate --split test
```

### 特征基线（推荐用于小样本）

从 skeleton + head_turn summary 提取特征：

```bash
python scripts/extract_spa_mmd_features.py \
  --dataset_root F:/SpA-MMD/processed \
  --sessions walk,head_turn \
  --output_csv D:/projects/SpA-GaitFormer/baselines/spa_mmd_features.csv
```

训练传统基线分类器（二分类任务）：

```bash
python scripts/train_baseline_classifier.py \
  --features_csv D:/projects/SpA-GaitFormer/baselines/spa_mmd_features.csv \
  --train_split F:/SpA-MMD/splits/train_subjects.txt \
  --test_split F:/SpA-MMD/splits/test_subjects.txt \
  --task disease \
  --model logreg
```

严重程度分级：

```bash
python scripts/train_baseline_classifier.py \
  --features_csv D:/projects/SpA-GaitFormer/baselines/spa_mmd_features.csv \
  --train_split F:/SpA-MMD/splits/train_subjects.txt \
  --test_split F:/SpA-MMD/splits/test_subjects.txt \
  --task severity \
  --model svm
```

### 滑窗训练（可选）

生成滑窗清单（保持受试者隔离）：

```bash
python scripts/build_window_manifest.py \
  --processed_root F:/SpA-MMD/processed \
  --split_file F:/SpA-MMD/splits/train_subjects.txt \
  --sessions walk,head_turn \
  --window 32 \
  --stride 16 \
  --output_csv D:/projects/SpA-GaitFormer/windows/train_windows.csv
```

然后在配置里设置 `data.train_window_manifest`（可选再设 test/val）：

```json
"train_window_manifest": "D:/projects/SpA-GaitFormer/windows/train_windows.csv"
```

当提供 window manifest 时，数据加载器会先按 `start_frame/end_frame` 切片，
再采样到 `data.num_frames`。

### 输出目录

交叉验证结果保存在：

```text
checkpoints/spa_mmd_dual_stream/cv/
```

最终训练结果保存在：

```text
checkpoints/spa_mmd_dual_stream/final/
```

### 建议汇报指标

疾病二分类：

- accuracy
- precision
- recall
- F1

严重程度预测：

- severity accuracy
- severity macro-F1
