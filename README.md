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

The current implementation directly targets the SpA-MMD `processed/subject/session` structure and uses the gait task by default:

```text
SpA-MMD/
└── processed/
    ├── S01/
    │   └── walk/
    │       ├── rgb/
    │       ├── labels/
    │       └── skeleton/
    │           └── kpt3d/
    │               └── kpt3d.npy
    └── ...
```

Default inputs:

- RGB: `walk/rgb/*.png`
- Skeleton: `walk/skeleton/kpt3d/kpt3d.npy`
- Binary label: `walk/labels/binary_label.txt`
- Severity label: `walk/labels/severity_label.txt`

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
- `model.input_mode`: `fusion`, `rgb`, or `skeleton`
- `cv.num_folds`: default `2`
- `cv.selection_metric`: default `loss`

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

当前实现直接适配 SpA-MMD 的 `processed/subject/session` 结构，默认使用 gait 任务：

```text
SpA-MMD/
└── processed/
    ├── S01/
    │   └── walk/
    │       ├── rgb/
    │       ├── labels/
    │       └── skeleton/
    │           └── kpt3d/
    │               └── kpt3d.npy
    └── ...
```

默认读取：

- RGB：`walk/rgb/*.png`
- Skeleton：`walk/skeleton/kpt3d/kpt3d.npy`
- 二分类标签：`walk/labels/binary_label.txt`
- 严重程度标签：`walk/labels/severity_label.txt`

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
- `model.input_mode`：`fusion`、`rgb` 或 `skeleton`
- `cv.num_folds`：默认 `2`
- `cv.selection_metric`：默认 `loss`

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
