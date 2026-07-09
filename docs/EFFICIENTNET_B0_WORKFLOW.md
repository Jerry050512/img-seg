# EfficientNet-B0 开发与运行流程

C 负责人当前提交范围包括 2D EfficientNet-B0 轻量模型和统一批量推理接口。所有命令通过 `uv` 运行。

## 训练

```powershell
uv run imgseg-efficientnet train --config configs/efficientnet_b0/base.yaml
```

默认读取 `dataset/`，使用 `configs/data/split_seed42.yaml` 做 case-level split，并按 `configs/efficientnet_b0/base.yaml` 中的 `mask_overrides` 处理重复 mask。训练样本按 z 轴切片，mask 统一执行 `mask > 0` 二值化。

默认 checkpoint：

```text
checkpoints/efficientnet_b0/best.pt
```

## 评估

```powershell
uv run imgseg-efficientnet evaluate `
  --config configs/efficientnet_b0/base.yaml `
  --checkpoint checkpoints/efficientnet_b0/best.pt `
  --split test `
  --output-dir outputs/efficientnet_b0 `
  --output-json outputs/efficientnet_b0/metrics.json
```

评估会整例重建 NIfTI mask，并用公共 Dice、IoU、Precision、Recall 指标计算结果。

## 批量推理

```powershell
uv run imgseg-predict `
  --model efficientnet_b0 `
  --checkpoint checkpoints/efficientnet_b0/best.pt `
  --input-dir Test `
  --output-dir Test_Seg
```

输入支持：

- NIfTI：`*.nii`、`*.nii.gz`，输出 `<case>_Seg.nii.gz`。
- 2D 图片：`*.jpg`、`*.png`、`*.bmp`、`*.tif`，输出同名二值 `.png`。

## 当前限制

- 当前默认 `encoder_weights: imagenet`；离线环境如果无法取得预训练权重，会自动回退到随机初始化。
- 当前机器若没有 CUDA，只建议跑单元测试和小样本 smoke test；正式指标应在 GPU 机器完成训练后补入 `docs/EXPERIMENTS.md`。
