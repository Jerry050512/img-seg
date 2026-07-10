# EfficientNet-B0 开发与运行流程

C 负责人当前提交范围包括 2D U-Net + EfficientNet-B0 encoder 轻量模型，以及与 nnU-Net v2 共用的批量推理和 WebUI 接口。所有命令通过 `uv` 运行。

## 训练

以下两种 `--config` 位置都兼容；推荐使用第一种，与 README 中的示例保持一致：

```powershell
uv run imgseg-efficientnet train --config configs/efficientnet_b0/base.yaml
uv run imgseg-efficientnet --config configs/efficientnet_b0/base.yaml train
```

每次训练会生成一个微秒精度的 UTC `run_name`，格式为 `YYYYMMDDTHHMMSSffffffZ`，例如 `20260710T083015123456Z`。默认产物分别为：

- `checkpoints/efficientnet_b0/<run_name>/best.pth`
- `outputs/efficientnet_b0/<run_name>/train.log`

最佳权重采用原子替换保存。日志采用每行一个 JSON 对象的 JSONL 格式，记录训练开始、逐轮 loss/验证指标、最佳权重保存和训练完成事件。首条 `run_started` 还会记录以下主要超参数：模型架构与 encoder、输入通道和类别数、图像尺寸和切片采样比例、随机种子、epoch、batch size、优化器、学习率、weight decay、loss、AMP 请求值与实际启用值、设备、worker 数、验证阈值和指标聚合口径。

`--log-file <path>` 可修改日志的父目录和文件名模板，时间戳目录仍会自动插入。使用 `--output-dir <dir>` 时，checkpoint 和未单独覆盖的日志会共同写入 `<dir>/<run_name>/`：

```powershell
uv run imgseg-efficientnet train `
  --config configs/efficientnet_b0/base.yaml `
  --output-dir outputs/efficientnet_b0/runs
```

默认读取 `dataset/`，使用 `configs/data/split_seed42.yaml` 做 case-level split，并按 `configs/efficientnet_b0/base.yaml` 中的 `data.mask_overrides` 处理重复 mask。训练样本按 z 轴切片，mask 统一执行 `mask > 0` 二值化；只有训练集会根据 keep ratio 下采样空 slice，验证集保留每个验证 case 的全部 slice。

每轮验证先在同一 case 内累计全部 slice 的 TP、FP、FN、TN，再计算该 case 的 Dice、IoU、Precision、Recall，最后对 case 做宏平均。这样不会把 slice 当成彼此独立的数据划分，也不会让大量全空 slice 直接主导 best checkpoint 的选择。训练 AMP 只会在 CUDA 设备上启用。

## 配置项

主要配置位于 `configs/efficientnet_b0/base.yaml`：

| key | 默认值 | 用途 |
|---|---:|---|
| `data.image_size` | `[512, 512]` | 模型输入切片尺寸；预测后按最近邻插值还原 mask 尺寸。 |
| `data.foreground_slice_keep_ratio` | `1.0` | 训练集前景 slice 保留比例。 |
| `data.empty_slice_keep_ratio` | `0.25` | 训练集空 slice 保留比例；不作用于验证/测试。 |
| `model.encoder_weights` | `imagenet` | 仅用于训练初始化；加载完整 checkpoint 做评估/推理时不重复加载预训练权重。 |
| `training.batch_size` | `8` | 训练与验证 DataLoader batch size。 |
| `training.amp` | `true` | CUDA 训练时使用自动混合精度；CPU/MPS 上自动关闭。 |
| `training.learning_rate` | `0.0003` | AdamW 学习率。 |
| `training.weight_decay` | `0.01` | AdamW weight decay。 |
| `training.log_file` | `outputs/efficientnet_b0/train.log` | 日志路径模板；实际路径会在父目录与文件名之间插入时间戳目录。 |
| `runtime.device` | `auto` | 依次自动选择 CUDA、MPS、CPU，也可显式指定。 |
| `runtime.num_workers` | `0` | DataLoader worker 数；大于 0 时启用 persistent workers。 |
| `runtime.nifti_cache_size` | `8` | 每个 Dataset/worker 缓存的 NIfTI image-mask proxy 对数量；设为 `0` 可关闭。 |
| `checkpoints.output_dir` | `checkpoints/efficientnet_b0` | WebUI 搜索 `.pth` 权重的目录。 |
| `checkpoints.best` | `checkpoints/efficientnet_b0/best.pth` | checkpoint 路径模板；实际路径会插入时间戳目录，且必须使用 `.pth` 后缀。 |
| `inference.threshold` | `0.5` | sigmoid 概率二值化阈值。 |
| `inference.batch_size` | `8` | 整卷推理时一次 forward 的 slice 数。 |
| `inference.amp` | `true` | CUDA 推理时使用自动混合精度；CPU/MPS 上自动关闭。 |
| `inference.output_dir` | `outputs/efficientnet_b0` | `evaluate` 未显式传输出目录时的预测目录。 |

`runtime.nifti_cache_size` 缓存的是 nibabel proxy/文件句柄，而不是把整卷影像全部读入内存；采用有界 LRU，DataLoader worker 启动时不会继承父进程中已打开的 proxy。其目标是减少逐 slice 重复 `nib.load` 的开销，实际吞吐和内存占用仍应在目标机器上测量。

## Checkpoint 与外部交付

默认 best checkpoint：

```text
checkpoints/efficientnet_b0/<run_name>/best.pth
```

`checkpoints/` 按项目规则不提交到 Git，因此仓库本身不提供可直接推理的 EfficientNet-B0 权重。模型负责人交付时必须同时提供 checkpoint 的外部存储位置或文件名、对应配置版本、验证/测试指标和简短实验结论。使用方需要先取得兼容的 `.pth` 文件，然后：

- 在 CLI 中通过 `--checkpoint <path>` 显式指定；或
- 将 `.pth` 放入 `checkpoints.output_dir`，供 WebUI 权重下拉框发现；或
- 在 WebUI 中手动填写 checkpoint 路径。

项目生成的 checkpoint 会保存模型与预处理配置；加载时会校验架构、encoder、通道数、`image_size` 和 `slice_axis`。若当前配置不兼容会直接报错，避免在错误预处理下静默生成结果。只有纯 state dict 等不含配置元数据的旧权重无法执行这项校验。

若自定义 `checkpoints.best`，建议让模板位于 `checkpoints.output_dir` 内。训练会把时间戳目录插入模板的父目录与文件名之间；WebUI 会递归查找该输出目录中的 `.pth`，并优先列出较新的时间戳运行。

## 评估

```powershell
uv run imgseg-efficientnet evaluate `
  --config configs/efficientnet_b0/base.yaml `
  --checkpoint checkpoints/efficientnet_b0/20260710T083015123456Z/best.pth `
  --split test `
  --output-dir outputs/efficientnet_b0 `
  --output-json outputs/efficientnet_b0/metrics.json
```

评估会逐 case 重建完整 NIfTI 二值 mask，保留参考影像的 affine/header，并使用公共 Dice、IoU、Precision、Recall 指标计算每例结果及宏平均，同时记录整例推理耗时。

## 正式实验结果（2026-07-10）

在带 NVIDIA GPU 的 PC 上使用默认配置完成 100 轮训练，ImageNet encoder 初始化成功。最佳权重保存于第 3 轮，对应验证集宏平均 Dice 0.7916。该次旧实验产物名为 `best.pt`；当前训练流程统一保存为 `<UTC时间戳>/best.pth`。随后使用上述评估命令在 `split_seed42` test split 的 2 个 case 上得到：

| Dice | IoU | Precision | Recall | 端到端 sec/case |
|---:|---:|---:|---:|---:|
| 0.9065 | 0.8313 | 0.9281 | 0.8878 | 11.16 |

这里的测试指标和耗时均为 case 宏平均。端到端耗时包含 NIfTI 读取、整卷 slice batching 推理和 mask 保存，不包含后续指标计算。逐 case 结果、训练曲线口径和实验结论见 `docs/EXPERIMENTS.md`；该次旧实验原始记录为 `outputs/efficientnet_b0/log.json` 和 `outputs/efficientnet_b0/metrics.json`。当前流程的逐轮记录默认写入 `outputs/efficientnet_b0/<UTC时间戳>/train.log`，评估结果仍可通过 `--output-json` 保存。

## 批量推理

```powershell
uv run imgseg-predict `
  --model efficientnet_b0 `
  --config configs/efficientnet_b0/base.yaml `
  --checkpoint checkpoints/efficientnet_b0/20260710T083015123456Z/best.pth `
  --input-dir Test `
  --output-dir Test_Seg
```

该 CLI 和 WebUI 调用同一公共批量推理接口。输入支持单个文件或递归扫描目录：

- NIfTI：`*.nii`、`*.nii.gz`，输出 `<case>_Seg.nii.gz`。
- 2D 图片：`*.jpg`、`*.jpeg`、`*.png`、`*.bmp`、`*.tif`、`*.tiff`，按灰度图直接预测，输出同名二值 `.png`。

整卷 NIfTI 不再逐 slice 单独 forward，而是按 `inference.batch_size` 分批预测；CUDA 上可由 `inference.amp` 启用推理 AMP。调大 batch size 通常会提高吞吐但增加显存占用，需要按目标 GPU 调整。

## WebUI

```powershell
uv run imgseg-webui
```

模型下拉框可选择 nnU-Net v2 或 EfficientNet-B0。切换模型后，WebUI 会读取对应配置并刷新 checkpoint 列表；输入收集、进度回报、二值结果统计和预览继续复用公共模块，WebUI 不直接依赖 EfficientNet 内部训练实现。

## 当前限制与实验状态

- 已完成一次 NVIDIA GPU 正式训练与 test split 评估；该次旧实验产物名为 `best.pt`，当前训练代码统一交付命名为 `checkpoints/efficientnet_b0/<UTC时间戳>/best.pth`。该路径仅是本地交付命名；文件不随 Git 分发，仍需上传外部存储后才能交付给其他负责人。
- 当前测试集只有 2 个 case，且两例 Dice 分别为 0.8674 和 0.9456；宏平均结果用于固定划分下的项目基线，不代表已充分验证跨数据泛化能力。
- 最佳验证 Dice 出现在第 3 轮，而训练完成到第 100 轮，后续实验应考虑 early stopping，并验证数据增强或正则化是否能缩小训练与验证趋势差异。
- 默认 `model.encoder_weights: imagenet`。训练初始化若无法取得预训练权重，会发出警告并回退到随机初始化；这会改变训练起点，必须在实验记录中说明。
- 该模型按 2D slice 建模，缺少显式 3D 上下文；边界连续性与泛化能力需要通过正式验证/测试确认。
- 没有 CUDA 时仍可运行 CPU/MPS smoke test，但 AMP 会自动关闭；不同硬件上的耗时不可与本次 NVIDIA GPU PC 结果直接比较。
