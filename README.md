# SpA-GaitFormer

Transformer-based dual-modal training pipeline for SpA-MMD, targeting:

- `healthy` / `SpA` binary classification
- disease severity prediction
- `RGB + skeleton(kpt3d)` multimodal fusion

This project was authored on macOS, but the implementation relies on Python, PyTorch, and `pathlib`, so it is intended to run on both Windows and Linux.

基于 Transformer 的 SpA-MMD 双模态训练工程，面向：

- `healthy` / `SpA` 二分类
- 患病严重程度预测
- `RGB + skeleton(kpt3d)` 双模态融合

这套代码在 macOS 上编写，但实现方式只依赖 Python、PyTorch 和 `pathlib`，训练时兼容 Windows 和 Linux。

## Data Assumptions

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

If your skeleton source is not `kpt3d.npy`, update `data.skeleton_file` and `joint_dim` in the config.

## 数据假设

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

如果你的骨架不是 `kpt3d.npy`，只需要改配置里的 `data.skeleton_file` 和 `joint_dim`。

## Project Structure

- [main.py](/Users/ice/Documents/SpA-GaitFormer/main.py): train/evaluate entry point
- [spa_gaitformer/data.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/data.py): SpA-MMD scanning, label loading, dual-modal input pipeline
- [spa_gaitformer/model.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/model.py): RGB, skeleton, and fusion Transformers
- [spa_gaitformer/losses.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/losses.py): multitask loss
- [spa_gaitformer/metrics.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/metrics.py): classification and severity metrics
- [spa_gaitformer/train.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/train.py): training, validation, checkpoints
- [configs/spa_mmd_dual_stream.json](/Users/ice/Documents/SpA-GaitFormer/configs/spa_mmd_dual_stream.json): config template
- [splits/train_subjects.txt](/Users/ice/Documents/SpA-GaitFormer/splits/train_subjects.txt): training subject list example
- [splits/val_subjects.txt](/Users/ice/Documents/SpA-GaitFormer/splits/val_subjects.txt): validation subject list example
- [splits/test_subjects.txt](/Users/ice/Documents/SpA-GaitFormer/splits/test_subjects.txt): test subject list example

## 项目结构

- [main.py](/Users/ice/Documents/SpA-GaitFormer/main.py)：训练/评估入口
- [spa_gaitformer/data.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/data.py)：SpA-MMD 目录扫描、标签读取、双模态加载
- [spa_gaitformer/model.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/model.py)：RGB、Skeleton、Fusion 三段 Transformer
- [spa_gaitformer/losses.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/losses.py)：多任务损失
- [spa_gaitformer/metrics.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/metrics.py)：分类和严重度指标
- [spa_gaitformer/train.py](/Users/ice/Documents/SpA-GaitFormer/spa_gaitformer/train.py)：训练、验证、checkpoint
- [configs/spa_mmd_dual_stream.json](/Users/ice/Documents/SpA-GaitFormer/configs/spa_mmd_dual_stream.json)：配置模板
- [splits/train_subjects.txt](/Users/ice/Documents/SpA-GaitFormer/splits/train_subjects.txt)：训练受试者列表示例
- [splits/val_subjects.txt](/Users/ice/Documents/SpA-GaitFormer/splits/val_subjects.txt)：验证受试者列表示例
- [splits/test_subjects.txt](/Users/ice/Documents/SpA-GaitFormer/splits/test_subjects.txt)：测试受试者列表示例

## Split File Format

Each split file contains one subject per line. By default, the loader appends `walk` automatically:

```text
S01
S02
S03
```

You can also specify the session explicitly:

```text
S01/walk
S02/walk
```

## Split 文件格式

每个 split 文件一行一个受试者，默认自动拼接 `walk`：

```text
S01
S02
S03
```

也可以显式写 session：

```text
S01/walk
S02/walk
```

## Configuration

All paths in the config are resolved relative to the config file location, so the same file can be adapted on Windows or Linux with either absolute or relative paths.

Key fields:

- `data.dataset_root`: root `processed` directory
- `data.train_split` / `val_split` / `test_split`: subject lists
- `data.session`: default `walk`
- `data.skeleton_file`: default `skeleton/kpt3d/kpt3d.npy`
- `data.max_joints`: number of joints
- `data.joint_dim`: use `3` for `kpt3d`, `2` for `kpt2d`
- `task.severity_mode`: `classification` or `regression`

## 配置

配置文件中的路径会相对配置文件位置解析，因此在 Windows 和 Linux 上都可以直接改成本机路径或保留相对路径。

关键字段：

- `data.dataset_root`：`processed` 根目录
- `data.train_split` / `val_split` / `test_split`：受试者列表
- `data.session`：默认 `walk`
- `data.skeleton_file`：默认 `skeleton/kpt3d/kpt3d.npy`
- `data.max_joints`：关节点数量
- `data.joint_dim`：`kpt3d` 用 `3`，`kpt2d` 用 `2`
- `task.severity_mode`：`classification` 或 `regression`

## Installation

Windows:

```powershell
python -m pip install -r requirements.txt
```

Linux:

```bash
python -m pip install -r requirements.txt
```

If your Linux environment only provides `python3`, replace `python` with `python3`.

## 安装

Windows:

```powershell
python -m pip install -r requirements.txt
```

Linux:

```bash
python -m pip install -r requirements.txt
```

如果你的 Linux 环境只有 `python3`，把命令里的 `python` 替换成 `python3` 即可。

## Training

Windows:

```powershell
python main.py --config configs/spa_mmd_dual_stream.json --mode train
```

Linux:

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode train
```

## 训练

Windows:

```powershell
python main.py --config configs/spa_mmd_dual_stream.json --mode train
```

Linux:

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode train
```

## Evaluation

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode evaluate --split val
```

Specify a checkpoint:

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode evaluate --split test --checkpoint checkpoints/spa_mmd_dual_stream/best.pt
```

## 评估

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode evaluate --split val
```

指定 checkpoint：

```bash
python main.py --config configs/spa_mmd_dual_stream.json --mode evaluate --split test --checkpoint checkpoints/spa_mmd_dual_stream/best.pt
```

## Manifest Export

If you want to export a split to CSV on the offline training machine for inspection, use:

```bash
python scripts/export_spa_mmd_manifest.py --dataset-root /path/to/processed --split-file splits/train_subjects.txt --output manifests/train.csv
```

## 导出 Manifest

如果你想在离线训练机上先把 split 导出成 CSV 供检查，可以用：

```bash
python scripts/export_spa_mmd_manifest.py --dataset-root /path/to/processed --split-file splits/train_subjects.txt --output manifests/train.csv
```

## Current Model

The current baseline is a three-stage Transformer:

1. RGB branch: per-frame patch embedding followed by spatial and temporal Transformers.
2. Skeleton branch: per-frame joint projection followed by joint-level and temporal Transformers.
3. Fusion branch: concatenates both temporal token streams and uses the fusion Transformer `CLS` token for disease classification and severity prediction.

## 当前模型

当前 baseline 是三段式 Transformer：

1. RGB 分支：每帧 patch embedding，随后做空间 Transformer 和时间 Transformer。
2. Skeleton 分支：每帧关节点投影后做 joint Transformer 和时间 Transformer。
3. 融合分支：拼接两路时序 token，用融合 Transformer 的 `CLS` token 同时输出疾病分类和严重程度预测。

## Notes Before Offline Training

- Confirm whether `kpt3d.npy` is shaped as `[T, J, 3]`; if not, the loader should be adjusted accordingly.
- Confirm whether `severity_label.txt` is categorical or continuous; if it is ordinal, an ordinal loss is preferable.
- Train/val/test must be split by subject rather than by session to avoid identity leakage.

## 离线训练前建议确认

- `kpt3d.npy` 的形状是否是 `[T, J, 3]`；如果不是，我再按真实形状改 loader。
- `severity_label.txt` 是离散等级还是连续分数；如果是有序等级，后续建议改成 ordinal loss。
- 训练/验证/测试必须按受试者划分，不能按 session 随机拆分，否则会有主体泄漏。
