# 数据集盘点

盘点时间：2026-07-10。当前 `dataset/` 下共有 13 个规范化原始样本目录，
每个目录均只保留一对 `image.nii.gz` 与 `mask.nii.gz`。布局规则配置为
`configs/data/dataset.yaml`。

旧原始目录已移动备份到：

```text
C:\tmp\img_seg_raw_before_standardize_20260710_101836
```

## 当前原始数据结构

```text
dataset/<case_id>/
  image.nii.gz
  mask.nii.gz
```

规范化规则：

- case id 统一为 ASCII 字母、数字和下划线。
- 图像与 mask 均统一压缩为 `.nii.gz`。
- mask 均为 `uint8`，标签集合均为 `0,1`。
- `S_1` 原始来源中的 `65535` 标签已按二分类任务转为 `1`。
- `nose_layer36` 原始来源中的标签 `2` 已按二分类任务转为 `1`。
- 所有样本 spacing 当前均为 1.0x1.0x1.0。

## 样本清单

| case_id | image | mask | shape | mask labels | foreground |
|---|---|---|---:|---|---:|
| `1_23_XY` | `image.nii.gz` | `mask.nii.gz` | 973x973x312 | 0,1 | 32.9006% |
| `2_25_XY` | `image.nii.gz` | `mask.nii.gz` | 973x973x298 | 0,1 | 40.9696% |
| `H1_20Layer` | `image.nii.gz` | `mask.nii.gz` | 512x512x351 | 0,1 | 17.1682% |
| `H2_4Layer` | `image.nii.gz` | `mask.nii.gz` | 512x512x281 | 0,1 | 17.6148% |
| `nose_layer4` | `image.nii.gz` | `mask.nii.gz` | 1317x914x156 | 0,1 | 29.7497% |
| `nose_layer12` | `image.nii.gz` | `mask.nii.gz` | 1319x914x245 | 0,1 | 44.3694% |
| `nose_layer20` | `image.nii.gz` | `mask.nii.gz` | 1316x913x271 | 0,1 | 25.4778% |
| `nose_layer28` | `image.nii.gz` | `mask.nii.gz` | 1312x912x194 | 0,1 | 9.9428% |
| `nose_layer36` | `image.nii.gz` | `mask.nii.gz` | 516x910x179 | 0,1 | 11.3416% |
| `S_1` | `image.nii.gz` | `mask.nii.gz` | 512x512x216 | 0,1 | 44.4292% |
| `S_3` | `image.nii.gz` | `mask.nii.gz` | 512x512x291 | 0,1 | 42.7604% |
| `S_4` | `image.nii.gz` | `mask.nii.gz` | 512x512x331 | 0,1 | 46.0658% |
| `S_5` | `image.nii.gz` | `mask.nii.gz` | 512x512x331 | 0,1 | 44.9559% |

## 训练读取入口

三个模型配置均指向同一份数据布局配置：

```text
configs/data/dataset.yaml
```

nnU-Net v2 配置：

```yaml
paths:
  dataset_file: configs/data/dataset.yaml
  raw_dataset_dir: dataset
  nnunet_raw: dataset/processed/nnunet_raw
  nnunet_preprocessed: dataset/processed/nnunet_preprocessed
```

MONAI SegResNet 与 EfficientNet-B0 配置：

```yaml
data:
  dataset_config: configs/data/dataset.yaml
  split_file: configs/data/split_seed42.yaml
```

`prepare_nnunet_dataset()` 会优先读取 `paths.dataset_file`，并在转换前校验每例
image/mask 的 shape 与 affine。`dataset.yaml` 当前使用 `layout: case_dir_pair`
自动扫描 `dataset/<case_id>/image.nii.gz` 与 `mask.nii.gz`。代码仍兼容显式
`cases:` 清单，用于数据尚未规范化、需要逐例指定路径的阶段。

## 当前划分

由 `configs/data/split_seed42.yaml` 生成，当前 13 例划分为 train=9、val=2、test=2。

| split | cases |
|---|---|
| train | `H1_20Layer`, `H2_4Layer`, `S_1`, `S_4`, `S_5`, `nose_layer12`, `nose_layer20`, `nose_layer28`, `nose_layer36` |
| val | `1_23_XY`, `nose_layer4` |
| test | `2_25_XY`, `S_3` |

nnU-Net 的 `numTraining` 为 11，即 train + val；test mask 同步输出到 `labelsTs` 用于独立评估。

## processed 验证结果

已重新生成：

```powershell
uv run imgseg-nnunet --config configs/nnunet_v2/base.yaml prepare
```

输出结构：

```text
dataset/processed/nnunet_raw/Dataset501_ImgSeg/
  dataset.json
  img_seg_split.json
  imagesTr/<case_id>_0000.nii.gz
  labelsTr/<case_id>.nii.gz
  imagesTs/<case_id>_0000.nii.gz
  labelsTs/<case_id>.nii.gz

dataset/processed/nnunet_preprocessed/Dataset501_ImgSeg/
  splits_final.json
```

验证命令：

```powershell
uv run python utils/audit_nifti.py dataset\processed\nnunet_raw\Dataset501_ImgSeg --labels
```

验证结果：

- `dataset/processed/nnunet_raw/Dataset501_ImgSeg` 已生成 13 个 image 和 13 个 label。
- 所有 processed label 均为 `uint8`。
- 所有 processed label 的标签集合均为 `0,1`。
- `2_25_XY` processed test label 前景比例为 40.9696%，与当前规范原始 mask 一致。
- `S_1` processed train label 前景比例为 44.4292%，与当前规范原始 mask 一致。

## 维护建议

- 后续新增样本直接按 `dataset/<case_id>/image.nii.gz` 与 `mask.nii.gz` 放置，不需要修改 `dataset.yaml`。
- 不要手工把同一 case 的多个版本放回主数据目录；需要保留历史版本时放在数据集目录之外或单独备份。
- 合并前重新运行 `uv run imgseg-nnunet --config configs/nnunet_v2/base.yaml prepare` 和相关 smoke test。
