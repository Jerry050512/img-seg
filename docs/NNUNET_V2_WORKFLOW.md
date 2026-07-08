# nnU-Net v2 开发与运行流程

本分支提供 nnU-Net v2 的准备、训练、推理和评估链路。所有命令通过 `uv` 在项目虚拟环境中运行。

## 1. 同步环境

```powershell
uv sync
```

如果当前 shell 找不到 `uv`，本机可使用完整路径：

```powershell
& "C:\Users\Jerry\AppData\Local\Microsoft\WinGet\Links\uv.exe" sync
```

## 2. 转换数据

```powershell
uv run imgseg-nnunet --config configs/nnunet_v2/base.yaml prepare
```

该命令会：

- 扫描 `dataset/` 下每个 case 目录。
- 根据 `configs/nnunet_v2/base.yaml` 中的 `dataset.mask_overrides` 解决重复 mask。
- 将训练/验证 case 写入 `dataset/processed/nnunet_raw/Dataset501_ImgSeg/imagesTr` 与 `labelsTr`。
- 将测试 case 写入 `imagesTs`，测试标签另存到 `labelsTs` 供项目评估使用。
- 将所有 mask 统一转换为二值 `0/1`，并输出 `.nii.gz`。

## 3. 规划和预处理

```powershell
uv run imgseg-nnunet plan
```

等价于调用：

```powershell
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

项目会自动设置：

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

## 4. 训练

RTX 4060 8GB 建议先训练 2D 配置：

```powershell
uv run imgseg-nnunet train --configuration 2d --fold 0
```

如果显存和时间允许，再尝试：

```powershell
uv run imgseg-nnunet train --configuration 3d_lowres --fold 0
```

## 5. 推理

默认对转换后的 `imagesTs` 推理：

```powershell
uv run imgseg-nnunet predict --configuration 2d --folds 0
```

也可以指定输入输出目录：

```powershell
uv run imgseg-nnunet predict --input-dir Test --output-dir Test_Seg --configuration 2d --folds 0
```

## 6. 评估

```powershell
uv run imgseg-nnunet evaluate `
  --prediction-dir outputs/nnunet_v2 `
  --reference-dir dataset/processed/nnunet_raw/Dataset501_ImgSeg/labelsTs `
  --output-json outputs/nnunet_v2/metrics.json
```

当前公共指标包含 Dice、IoU、Precision、Recall，并保留 TP/FP/FN/TN 供报告分析。

## Smoke Test

不启动真实 nnU-Net 训练时，可以先检查命令拼接：

```powershell
uv run imgseg-nnunet plan --dry-run
uv run imgseg-nnunet train --configuration 2d --fold 0 --dry-run
uv run imgseg-nnunet predict --configuration 2d --folds 0 --dry-run
```

## 本机验证记录

2026-07-08 在 RTX 4060 8GB 机器上已完成以下 smoke/集成验证：

- `uv sync` 成功创建 `.venv` 并安装依赖。
- `uv run pytest` 通过，当前 6 个测试全绿。
- `uv run ruff check .` 通过。
- `uv run imgseg-nnunet prepare` 已将真实 `dataset/` 转为 `Dataset501_ImgSeg`，划分为 train=6、val=1、test=1。
- `uv run imgseg-nnunet plan` 已通过 nnU-Net v2 官方 `verify_dataset_integrity`，并完成 fingerprint、plans 与 2d/3d_fullres/3d_lowres 预处理。
- 使用 `labelsTs` 对自身进行 identity evaluate，Dice/IoU/Precision/Recall 均为 1.0，确认评估链路能读取真实 NIfTI。

当前尚未启动真实训练；建议先从 `2d fold 0` 开始训练。
