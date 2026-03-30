# PET-CT Dual-Modal 3D Lesion Segmentation

本项目实现了一个 PET/CT 双模态 3D 病灶分割流程，包含：

- PET 5 级 3D CNN 编码器
- CT 官方 `SegMamba-V2` 分级编码器接入
- 逐尺度 PET/CT 融合
- 3D U 形解码器
- 预处理、训练、评估、推理、可视化、5-fold 交叉验证脚本

当前代码默认优先使用官方 `SegMamba-V2` 作为 CT 编码器；若依赖缺失，可回退到本地兼容版 `fallback_segmamba_style`。正式实验建议显式开启 `--disable-ct-fallback`，避免误用回退版本。

## 1. 项目结构

```text
.
|-- preprocess_pet_ct_dataset.py
|-- normalize_petct_raw_names.py
|-- train.py
|-- evaluate.py
|-- infer.py
|-- crossval.py
|-- visualize_cases.py
|-- smoke_test.py
|-- petct_seg_model.py
|-- petct_official_segmamba_v2.py
|-- petct_seg_data.py
|-- petct_seg_losses.py
|-- petct_seg_metrics.py
|-- SegMamba-V2-main/
`-- README.md
```

## 2. 环境配置

推荐环境：

- Linux
- NVIDIA GPU
- Python 3.10 或 3.11
- PyTorch + CUDA 与机器环境匹配

### 2.1 创建环境

使用 Conda：

```bash
conda create -n petct-seg python=3.10 -y
conda activate petct-seg
python -m pip install --upgrade pip setuptools wheel
```

或使用 venv：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2.2 安装 PyTorch

请先按你的 CUDA 版本安装对应的 PyTorch。安装后先检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 2.3 安装项目依赖

基础依赖：

```bash
pip install -r requirements.txt
```

如果要使用官方 `SegMamba-V2` CT 编码器，还需要：

```bash
pip install monai einops mamba-ssm
```

### 2.4 检查官方 CT 编码器是否可用

```bash
python -c "from petct_seg_model import DualModalSegNet3D; m = DualModalSegNet3D(); print(m.ct_encoder_type)"
```

期望输出：

```text
official_segmamba_v2
```

如果输出是 `fallback_segmamba_style`，说明官方依赖尚未正确加载。

### 2.5 运行最小自检

```bash
python smoke_test.py
```

这个脚本会检查：

- 前向传播是否正常
- 多尺度特征金字塔是否正常生成
- 损失与反向传播是否正常

## 3. 数据格式

### 3.1 通用原始数据格式

`preprocess_pet_ct_dataset.py` 期望输入目录为：

```text
raw_cases/
|-- case_001/
|   |-- ct.nii.gz
|   |-- pet.nii.gz
|   |-- ct_seg.nii.gz
|   `-- pet_seg.nii.gz
`-- case_002/
```

其中：

- `ct.nii.gz`: CT 体数据
- `pet.nii.gz`: PET 体数据
- `ct_seg.nii.gz`: CT 标注，可选
- `pet_seg.nii.gz`: PET 标注，可选

### 3.2 处理后数据格式

预处理输出目录为：

```text
processed_cases/
|-- case_001/
|   |-- ct.npy
|   |-- pet.npy
|   |-- mask.npy
|   |-- ct_label.npy
|   |-- pet_label.npy
|   |-- pet_to_ct.tfm
|   `-- metadata.json
`-- case_002/
```

训练和评估实际使用的是：

- `ct.npy`
- `pet.npy`
- `mask.npy`

### 3.3 如果原始文件名不规范

如果你的原始病例目录已经是扁平 case 结构，但文件名不是 `ct.nii.gz / pet.nii.gz / ct_seg.nii.gz / pet_seg.nii.gz`，可以先运行：

```bash
python normalize_petct_raw_names.py \
  --input-root ./raw_cases_original \
  --output-root ./raw_cases_normalized
```

## 4. 数据预处理

```bash
python preprocess_pet_ct_dataset.py \
  --input-root ./raw_cases_normalized \
  --output-root ./processed_cases \
  --target-spacing 2.0,2.0,2.0 \
  --crop-size 128,128,128 \
  --crop-mode auto \
  --label-mode union \
  --transform-type rigid
```

说明：

- `--crop-mode auto`: 有病灶时围绕病灶裁剪，否则围绕 body 裁剪
- `--label-mode union`: `ct_seg` 和 `pet_seg` 并集作为最终监督
- `--transform-type rigid`: 先做 PET 到 CT 的刚体配准

如果 PET 和 CT 已经严格对齐，可以跳过配准：

```bash
python preprocess_pet_ct_dataset.py \
  --input-root ./raw_cases_normalized \
  --output-root ./processed_cases \
  --target-spacing 2.0,2.0,2.0 \
  --crop-size 128,128,128 \
  --crop-mode auto \
  --label-mode union \
  --skip-registration
```

注意：

- `--allow-expand-crop` 会根据病灶尺寸扩展 patch，可能导致显存显著增加
- 如果 GPU 只有 24GB 左右，默认不建议在正式训练中盲目开启 `--allow-expand-crop`

## 5. 训练

### 5.1 `train.py` 的输入要求

`train.py` 需要你已经有单独的训练集和验证集目录，例如：

```text
data_split/
|-- train/
|   `-- case_xxx/
`-- val/
    `-- case_yyy/
```

如果你手里只有一个完整的 `processed_cases/`，建议直接用 `crossval.py` 负责划分。

### 5.2 单次训练

当前推荐配置：

- `channels=16,32,64,128,256`
- `pet_depths=1,1,1,1,1`
- `ct_depths=1,1,1,1`
- `ct_encoder_type=official_segmamba_v2`
- `loss=dice_bce`
- `dice_weight=1.0`
- `bce_weight=0.5`
- `bce_pos_weight=1.0`
- `batch_size=1`
- `amp`

训练命令：

```bash
python train.py \
  --train-dir ./data_split/train \
  --val-dir ./data_split/val \
  --save-dir ./checkpoints/run_final \
  --epochs 100 \
  --batch-size 1 \
  --channels 16,32,64,128,256 \
  --pet-depths 1,1,1,1,1 \
  --ct-depths 1,1,1,1 \
  --ct-encoder-type official_segmamba_v2 \
  --official-segmamba-path ./SegMamba-V2-main/brats23/models_segmamba/segmambav2.py \
  --disable-ct-fallback \
  --loss dice_bce \
  --dice-weight 1.0 \
  --bce-weight 0.5 \
  --bce-pos-weight 1.0 \
  --amp \
  --compute-val-hd95
```

训练结束后，`--save-dir` 下通常会生成：

- `best.pt`
- `last.pt`
- `run_config.json`

注意：

- 只有提供了 `--val-dir`，才会根据验证集指标写出 `best.pt`
- 若不提供 `--val-dir`，只有 `last.pt`

### 5.3 可选训练项

支持的损失函数：

- `dice_bce`
- `tversky`
- `focal_tversky`

支持的训练增强：

- 随机 3D flip
- 强度 scale / shift
- Gaussian noise

支持的采样策略：

- `uniform`
- `lesion_size_bins`

例如：

```bash
python train.py \
  --train-dir ./data_split/train \
  --val-dir ./data_split/val \
  --save-dir ./checkpoints/run_aug \
  --epochs 100 \
  --batch-size 1 \
  --channels 16,32,64,128,256 \
  --pet-depths 1,1,1,1,1 \
  --ct-depths 1,1,1,1 \
  --ct-encoder-type official_segmamba_v2 \
  --official-segmamba-path ./SegMamba-V2-main/brats23/models_segmamba/segmambav2.py \
  --disable-ct-fallback \
  --loss dice_bce \
  --dice-weight 1.0 \
  --bce-weight 0.5 \
  --bce-pos-weight 1.0 \
  --train-sampler lesion_size_bins \
  --small-case-weight 2.0 \
  --medium-case-weight 1.0 \
  --large-case-weight 0.7 \
  --augment \
  --amp \
  --compute-val-hd95
```

## 6. 评估

使用 `evaluate.py` 在指定数据集上评估 checkpoint：

```bash
python evaluate.py \
  --checkpoint ./checkpoints/run_final/best.pt \
  --dataset-dir ./data_split/test \
  --output-json ./eval/run_final_test.json \
  --save-preds-dir ./preds/run_final_test
```

输出包括：

- 每个病例的 Dice / HD95 / `gt_voxels` / `pred_voxels`
- 汇总 JSON
- 可选的预测概率图与二值 mask

## 7. 推理

对单个病例做推理：

```bash
python infer.py \
  --checkpoint ./checkpoints/run_final/best.pt \
  --pet ./demo/pet.npy \
  --ct ./demo/ct.npy \
  --output-prob ./outputs/prob.npy \
  --output-mask ./outputs/mask.npy \
  --threshold 0.5
```

## 8. 可视化

`visualize_cases.py` 可以从 checkpoint 直接推理，也可以读取 `evaluate.py` 已保存的预测结果。

### 8.1 可视化指定病例

```bash
python visualize_cases.py \
  --dataset-dir ./data_split/test \
  --preds-dir ./preds/run_final_test \
  --case-id case_001 \
  --output-dir ./viz/case_001
```

### 8.2 可视化最差的 3 个病例

```bash
python visualize_cases.py \
  --dataset-dir ./data_split/test \
  --preds-dir ./preds/run_final_test \
  --eval-json ./eval/run_final_test.json \
  --top-k-worst 3 \
  --output-dir ./viz/worst3
```

### 8.3 可视化最好的 3 个病例

```bash
python visualize_cases.py \
  --dataset-dir ./data_split/test \
  --preds-dir ./preds/run_final_test \
  --eval-json ./eval/run_final_test.json \
  --top-k-best 3 \
  --output-dir ./viz/best3
```

### 8.4 可视化 Dice 大于 0.9 的高分病例

```bash
python visualize_cases.py \
  --dataset-dir ./data_split/test \
  --preds-dir ./preds/run_final_test \
  --eval-json ./eval/run_final_test.json \
  --min-dice 0.9 \
  --output-dir ./viz/high_dice
```

如果没有任何病例满足 `Dice >= 0.9`，脚本会报：

```text
No cases matched the requested Dice filter.
```

这时可以降低阈值，或者改用 `--top-k-best`。

## 9. 5-fold 交叉验证

`crossval.py` 直接基于一个 `processed_cases/` 根目录自动切分 train/val/test，并串联 `train.py + evaluate.py`。

### 9.1 先只跑 1 折 sanity check

```bash
python crossval.py \
  --dataset-dir ./processed_cases \
  --output-dir ./cv_run \
  --num-folds 5 \
  --fold-indices 1 \
  --stratify-by mask_voxels \
  --split-mode auto \
  --eval-device cuda \
  -- \
  --epochs 100 \
  --batch-size 1 \
  --channels 16,32,64,128,256 \
  --pet-depths 1,1,1,1,1 \
  --ct-depths 1,1,1,1 \
  --ct-encoder-type official_segmamba_v2 \
  --official-segmamba-path ./SegMamba-V2-main/brats23/models_segmamba/segmambav2.py \
  --disable-ct-fallback \
  --loss dice_bce \
  --dice-weight 1.0 \
  --bce-weight 0.5 \
  --bce-pos-weight 1.0 \
  --amp \
  --compute-val-hd95
```

### 9.2 正式跑完整 5-fold

```bash
python crossval.py \
  --dataset-dir ./processed_cases \
  --output-dir ./cv_run \
  --num-folds 5 \
  --val-fraction 0.2 \
  --stratify-by mask_voxels \
  --split-mode auto \
  --eval-device cuda \
  --save-preds \
  -- \
  --epochs 100 \
  --batch-size 1 \
  --channels 16,32,64,128,256 \
  --pet-depths 1,1,1,1,1 \
  --ct-depths 1,1,1,1 \
  --ct-encoder-type official_segmamba_v2 \
  --official-segmamba-path ./SegMamba-V2-main/brats23/models_segmamba/segmambav2.py \
  --disable-ct-fallback \
  --loss dice_bce \
  --dice-weight 1.0 \
  --bce-weight 0.5 \
  --bce-pos-weight 1.0 \
  --amp \
  --compute-val-hd95
```

交叉验证结束后会生成：

- `crossval_manifest.json`
- `crossval_summary.json`
- 每折的 `split/`
- 每折的 `checkpoints/`
- 每折的 `eval/results.json`
- 可选的每折 `preds/`

## 10. 当前固定实验配置与结果

当前冻结的推荐实验配置是：

- `channels=16,32,64,128,256`
- `pet_depths=1,1,1,1,1`
- `ct_depths=1,1,1,1`
- `ct_encoder_type=official_segmamba_v2`
- `loss=dice_bce`
- `dice_weight=1.0`
- `bce_weight=0.5`
- `bce_pos_weight=1.0`
- `batch_size=1`
- `amp`

在既有 synthetic 数据上的正式 5-fold 结果为：

- `Dice = 0.6676 ± 0.0357`
- `HD95 = 15.3004 ± 3.8397`

对应 fold 级验证集均值：

- `val Dice mean = 0.6955`
- `val HD95 mean = 17.2629`

已验证但未优于该配置的尝试包括：

- 更深网络
- `tversky`
- `focal_tversky`
- 更高 `bce_pos_weight`
- 轻量增强
- lesion-size weighted sampling

## 11. 常见问题

### 11.1 训练时没有 `best.pt`

原因通常是没有传 `--val-dir`。

### 11.2 模型自动回退到了 `fallback_segmamba_style`

说明官方 `SegMamba-V2` 没有正确导入。检查：

```bash
python -c "import monai, einops, mamba_ssm; print('ok')"
```

同时检查：

- `--official-segmamba-path` 路径是否正确
- 当前 Python / CUDA / PyTorch / `mamba-ssm` 是否兼容

### 11.3 显存不足

优先检查：

- 预处理 patch 是否过大
- 是否使用了 `--allow-expand-crop`
- `channels` 是否过大

常见缓解方法：

- 去掉 `--allow-expand-crop`
- 减小 `crop-size`
- 改小 `channels`
- 保持 `batch-size=1`
- 开启 `--amp`

### 11.4 `visualize_cases.py --min-dice 0.9` 报错

这通常不是脚本坏了，而是评估 JSON 里确实没有任何病例满足 `Dice >= 0.9`。此时：

- 降低阈值，例如 `--min-dice 0.8`
- 或者改用 `--top-k-best 1`

## 12. 推荐执行顺序

建议按下面顺序完整跑实验：

1. 创建 Python 环境并安装 PyTorch
2. 安装 `requirements.txt`
3. 如需官方 CT 编码器，再安装 `monai / einops / mamba-ssm`
4. 运行 `python smoke_test.py`
5. 准备原始数据目录
6. 如有需要，先运行 `normalize_petct_raw_names.py`
7. 运行 `preprocess_pet_ct_dataset.py`
8. 先做 1 折 sanity check
9. 确认训练、评估、可视化结果正常后，再跑完整 5-fold
