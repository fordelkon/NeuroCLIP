# NeuroCLIP - 中文说明

EEG-to-CLIP 表征学习工作流，用于 THINGS-EEG2 数据集的预处理、特征提取、训练和评估。

## 核心流程

```text
环境配置 → 解压数据 → 预处理EEG → 提取CLIP特征 → 建立知识图谱 → 训练模型 → 评估模型
```

## 快速开始

### 1. 安装环境

```bash
# 安装 uv 包管理器
# Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# CPU 环境
uv sync --extra cpu

# GPU 环境 (CUDA 12.6)
uv sync --extra cu126
```

### 2. 下载数据集

从以下链接下载 THINGS-EEG2 数据集：

- **EEG 数据**: [Google Drive](https://drive.google.com/drive/folders/1KnOcV38RthPcpZR2vtiSm0jtZ6p63RNt)
- **训练图像**: [OSF](https://osf.io/y63gw/files/3v527)
- **测试图像**: [OSF](https://osf.io/y63gw/files/znu7b)

下载后按以下结构组织：
```text
<eeg_zip_dir>/
  sub-01.zip, sub-02.zip, ...
<image_zip_dir>/
  training_images.zip, test_images.zip
```

### 3. 配置路径（可选）

编辑 `configs/paths/default.yaml` 设置数据路径，或在命令行中使用 `paths.*` 参数覆盖。

默认路径：
- EEG 原始数据: `./data/thingseeg2-eegs`
- 图像数据: `./data/thingseeg2-images`
- 预处理数据: `./data/thingseeg2-eeg2-250hz`
- CLIP 特征: `./data/thingseeg2-clip-features`

### 4. 解压数据集

**Windows:**
```powershell
.\scripts\windows\unzip.ps1 <eeg_zip_dir> <image_zip_dir>
```

**Linux:**
```bash
bash scripts/linux/unzip.sh <eeg_zip_dir> <image_zip_dir>
```

**macOS:**
```bash
bash scripts/macos/unzip.sh <eeg_zip_dir> <image_zip_dir>
```

### 5. 预处理 EEG 数据

处理被试 1-10 的数据：

**Windows:**
```powershell
.\scripts\windows\preprocess.ps1 1 10
```

**Linux:**
```bash
bash scripts/linux/preprocess.sh 1 10
```

**macOS:**
```bash
bash scripts/macos/preprocess.sh 1 10
```

输出：为每个被试生成 `training.pt` 和 `test.pt` 文件。

### 6. 提取 CLIP 特征

**Windows:**
```powershell
.\scripts\windows\extract.ps1
```

**Linux:**
```bash
bash scripts/linux/extract.sh
```

**macOS:**
```bash
bash scripts/macos/extract.sh
```

输出：生成与 EEG 数据对齐的 CLIP 图像和文本特征。

### 7. 建立知识图谱

使用CLIP特征构建知识图谱，为后续的知识图谱增强训练做准备。

**Windows:**
```powershell
.\scripts\windows\build_kg.ps1
```

**Linux:**
```bash
bash scripts/linux/build_kg.sh
```

**macOS:**
```bash
bash scripts/macos/build_kg.sh
```

输出：生成知识图谱结构和相关数据文件。

### 8. 训练模型

根据需求选择不同的训练方式：

#### 8.1 基础训练（标准模型）

在 THINGS-EEG2 数据集上训练 NICE 模型（5折交叉验证）：

```bash
uv run --no-sync python src/train.py \
  data=thingseeg2 \
  data.k_fold=5 \
  data.fold_idx=1 \
  model=nice \
  model.loss_type=cliploss \
  trainer=gpu \
  trainer.max_epochs=100 \
  callbacks.early_stopping.patience=20
```

使用脚本快速训练：

**Windows:**
```powershell
.\scripts\windows\train.ps1
```

**Linux:**
```bash
bash scripts/linux/train.sh
```

**macOS:**
```bash
bash scripts/macos/train.sh
```

#### 8.2 知识图谱训练（KG增强模型）

使用知识图谱增强的模型训练（需先完成第7步）：

**Windows:**
```powershell
.\scripts\windows\train_kg.ps1
```

**Linux:**
```bash
bash scripts/linux/train_kg.sh
```

**macOS:**
```bash
bash scripts/macos/train_kg.sh
```

#### 8.3 并行训练（批量实验）

同时运行多个训练实验，适合超参数搜索或多折交叉验证：

**标准模型并行训练：**

**Windows:**
```powershell
.\scripts\windows\train_parallel.ps1
```

**Linux:**
```bash
bash scripts/linux/train_parallel.sh
```

**macOS:**
```bash
bash scripts/macos/train_parallel.sh
```

**知识图谱模型并行训练：**

**Windows:**
```powershell
.\scripts\windows\train_kg_parallel.ps1
```

**Linux:**
```bash
bash scripts/linux/train_kg_parallel.sh
```

**macOS:**
```bash
bash scripts/macos/train_kg_parallel.sh
```

### 9. 评估模型

使用训练好的检查点评估模型：

```bash
uv run --no-sync python src/eval.py \
  data=thingseeg2 \
  data.k_fold=5 \
  data.fold_idx=1 \
  model=nice \
  model.loss_type=cliploss \
  trainer=gpu \
  ckpt_path="/path/to/checkpoint.ckpt"
```

### 10. 运行测试

```bash
uv run --no-sync pytest
```

## 常用配置选项

### 数据配置

| 参数 | 说明 | 示例 |
|------|------|------|
| `data.subjects` | 选择被试 | `data.subjects=[sub-01,sub-02]` |
| `data.k_fold` | K折交叉验证 | `data.k_fold=5` |
| `data.fold_idx` | 选择折数（从0开始） | `data.fold_idx=0` |
| `data.train_batch_size` | 训练批次大小 | `data.train_batch_size=128` |
| `data.average_reps` | 平均重复试次 | `data.average_reps=true` |

### 模型配置

| 参数 | 说明 | 示例 |
|------|------|------|
| `model` | 选择模型 | `model=nice`, `model=atms` |
| `model.loss_type` | 损失函数类型 | `model.loss_type=cliploss` |
| `model.optimizer.lr` | 学习率 | `model.optimizer.lr=0.0001` |
| `model.optimizer.weight_decay` | 权重衰减 | `model.optimizer.weight_decay=0.0001` |

### 训练器配置

| 参数 | 说明 | 示例 |
|------|------|------|
| `trainer` | 训练器类型 | `trainer=cpu`, `trainer=gpu` |
| `trainer.max_epochs` | 最大训练轮数 | `trainer.max_epochs=100` |
| `trainer.devices` | 设备数量 | `trainer.devices=1` |
| `callbacks.early_stopping.patience` | 早停耐心值 | `callbacks.early_stopping.patience=20` |

## 预处理选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `preprocess.sfreq` | 目标采样频率 | 250 |
| `preprocess.mvnn_dim` | 归一化模式 | `time` |
| `preprocess.tmin`, `preprocess.tmax` | 时间窗口 | 0, 1 |

归一化模式：
- `time`: 时间维度MVNN白化
- `epoch`: 试次维度MVNN白化  
- `null`: 仅通道级z-score归一化

## CLIP 特征提取选项

| 参数 | 说明 | 示例 |
|------|------|------|
| `extract.model_name` | CLIP模型名称 | `ViT-B-32` |
| `extract.feature_mode` | 特征类型 | `pooled`, `last_hidden_state_no_cls` |
| `extract.device` | 推理设备 | `auto`, `cuda`, `cpu` |
| `extract.extract_text` | 是否提取文本特征 | `true`, `false` |

## 项目结构

```text
configs/          # Hydra 配置文件
scripts/          # 平台相关的脚本
  windows/        # Windows PowerShell 脚本
  linux/          # Linux Bash 脚本
  macos/          # macOS Bash 脚本
src/              # 源代码
  train.py        # 训练入口
  eval.py         # 评估入口
  preprocess.py   # 预处理入口
  extract.py      # 特征提取入口
  data/           # 数据模块
  models/         # 模型定义
tests/            # 测试文件
```

## 技术栈

- **运行时**: Python 3.10+, uv
- **深度学习**: PyTorch, Lightning
- **配置管理**: Hydra
- **EEG处理**: MNE, SciPy, scikit-learn
- **特征提取**: Transformers, Hugging Face CLIP

## 注意事项

1. 首次运行前确保使用 `uv sync` 同步环境
2. 后续运行使用 `uv run --no-sync` 跳过同步以提高速度
3. 使用 Hydra 命令行覆盖进行实验配置
4. 多设备训练时确保批次大小可被设备数整除
5. 保持训练和评估时的配置一致（data、model、k_fold等）

## 常见问题

**Q: 如何使用GPU训练？**  
A: 使用 `trainer=gpu` 或 `trainer=ddp trainer.devices=4` 进行分布式训练。

**Q: 如何恢复训练？**  
A: 添加 `ckpt_path=/path/to/last.ckpt` 参数。

**Q: 如何进行超参数搜索？**  
A: 使用 `-m` 标志和 `hparams_search` 配置：
```bash
uv run --no-sync python src/train.py -m \
  hparams_search=nice_optuna \
  experiment=nice_experiment \
  trainer=gpu
```

**Q: 预处理需要多长时间？**  
A: 单个被试约需几分钟，具体取决于硬件配置。

**Q: 训练需要多少显存？**  
A: NICE模型在默认配置下约需 4-8GB 显存。

## 更多信息

详细的参数说明和高级用法请参考英文版 [README.md](README.md)。
