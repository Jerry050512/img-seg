# MONAI SegResNet 工作流

本工作流对应项目 B 部分：使用固定 case-level 划分训练 3D MONAI SegResNet，
通过随机前景/背景 patch 控制显存，并使用滑窗推理输出与原始 NIfTI 几何一致的二值 mask。

## 实现范围

- 数据配置：`configs/monai_segresnet/base.yaml`
- 模型与统一适配器：`src/img_seg/models/monai_segresnet.py`
- 训练、验证、预测、评估：`src/img_seg/training/monai_segresnet.py`
- 命令行入口：`imgseg-segresnet`
- checkpoint：`checkpoints/monai_segresnet/best.pt` 和 `last.pt`
- 历史记录：`outputs/monai_segresnet/history.json`

训练和验证直接读取 `dataset/<case_id>/image.nii.gz` 与 `mask.nii.gz`，并复用
`configs/data/split_seed42.yaml`。mask 在加载后统一执行 `mask > 0`。

## 训练前检查

```powershell
uv sync
uv run imgseg-segresnet --config configs/monai_segresnet/base.yaml inspect
```

`inspect` 会打印训练、验证、测试 case，以及模型参数量和 3D ROI。必须确认划分与
`docs/DATASET_AUDIT.md` 一致后再开始训练。

## 训练

默认配置面向 RTX 4060 8 GB：ROI 为 `96x96x64`、batch size 为 1、每个训练体
采样 2 个 patch，并启用 AMP；每 10 个 epoch 做一次完整验证。Windows 下默认
`workers: 0`，避免共享内存错误。

```powershell
uv run imgseg-segresnet --config configs/monai_segresnet/base.yaml train `
  --output-json outputs/monai_segresnet/train_summary.json
```

系统内存充足时，可缓存确定性加载/归一化结果并提高验证窗口并行度：

```powershell
uv run imgseg-segresnet --config configs/monai_segresnet/base.yaml train `
  --cache-rate 1.0 --sw-batch-size 4
```

当前 13 例全部缓存约需 13 GiB 以上可用内存；内存不足时不要启用该选项。

先做短程 smoke run：

```powershell
uv run imgseg-segresnet --config configs/monai_segresnet/base.yaml train --epochs 1
```

从 `last.pt` 继续训练到配置中的总 epoch：

```powershell
uv run imgseg-segresnet --config configs/monai_segresnet/base.yaml train `
  --resume checkpoints/monai_segresnet/last.pt
```

显存不足时按顺序调整：

1. 将 ROI 改为 `80x80x48` 或 `64x64x48`，各维保持可被 8 整除。
2. 保持 batch size 为 1，并将 `samples_per_volume` 从 2 改为 1。
3. 保持 AMP 开启；不要先关闭 AMP。
4. 最后再减少 `init_filters`，因为这会改变网络结构并使已有 checkpoint 失效。

## 测试集推理与评估

使用验证 Dice 最佳的 checkpoint 对固定测试集推理：

```powershell
uv run imgseg-segresnet --config configs/monai_segresnet/base.yaml predict `
  --checkpoint checkpoints/monai_segresnet/best.pt `
  --split test `
  --output-dir outputs/monai_segresnet/test_predictions
```

计算公共二分类指标：

```powershell
uv run imgseg-segresnet --config configs/monai_segresnet/base.yaml evaluate `
  --prediction-dir outputs/monai_segresnet/test_predictions `
  --split test `
  --output-json outputs/monai_segresnet/test_metrics.json
```

输出包含逐 case 与平均 Dice、IoU、Precision、Recall，以及预测 manifest 中可用时的
平均单例推理耗时。最终将结果追加到 `docs/EXPERIMENTS.md`；checkpoint 文件不提交 Git，
只记录外部存储位置或约定文件名。

## Checkpoint 约定

`best.pt` 由验证集平均 Dice 选择，`last.pt` 每个 epoch 更新。二者均包含：

- `model_state`
- `optimizer_state`
- `scheduler_state`
- `scaler_state`
- `epoch`
- `best_dice`
- 模型配置与 ROI

修改 `blocks_down`、`blocks_up`、`init_filters` 或输入输出通道后，旧 checkpoint 不能
再以严格模式加载。只调整推理 overlap 或 threshold 不改变 checkpoint 兼容性。
