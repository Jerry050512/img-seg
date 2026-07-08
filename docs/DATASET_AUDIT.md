# 数据集盘点

盘点时间：2026-07-08。当前 `dataset/` 下共有 8 个样本目录，扫描图像与 mask 的空间尺寸均能一一对应，但命名与标签值尚未统一。

## 样本清单

| case | image | image shape | image dtype | mask | mask labels | foreground |
|---|---|---:|---|---|---|---:|
| `1_23_XY` | `1_23_XY.nii.gz` | 973x973x312 | uint8 | `1_23_XY.Seg.nii` | 0, 1 | 32.9006% |
| `1_23_XY` | `1_23_XY.nii.gz` | 973x973x312 | uint8 | `1_23_XY_Seg(1).nii` | 0, 1 | 29.9702% |
| `H1-20Layer` | `H1-20Layer.nii` | 512x512x351 | uint8 | `H1-20Layer_Seg.nii` | 0, 1 | 17.1682% |
| `H2-4Layer` | `H2-4Layer.nii` | 512x512x281 | uint8 | `H2-4Layer_Seg.nii` | 0, 1 | 17.6148% |
| `nose_layer28` | `nose_layer28.nii` | 1312x912x194 | uint8 | `nose_layer28_Seg.nii` | 0, 1 | 9.9428% |
| `nose_layer36` | `nose_layer36.nii` | 516x910x179 | uint8 | `nose_layer36_Seg(1)(1).nii` | 0, 1, 2 | 11.3416% |
| `nose_layer4` | `nose_layer4.nii` | 1317x914x156 | uint8 | `nose_layer4_Seg.nii` | 0, 1 | 29.7497% |
| `S-3` | `S-3.nii` | 512x512x291 | uint8 | `S-3_Seg.nii` | 0, 1 | 42.7604% |
| `S_1` | `S-1.nii` | 512x512x216 | uint8 | `S-1-Seg.nii` | 0, 1 | 43.2832% |
| `S_1` | `S-1.nii` | 512x512x216 | uint8 | `S_1_Seg.nii.gz` | 0, 65535 | 44.4292% |

全部样本 spacing 当前均为 1.0x1.0x1.0。

## 需要人工确认的问题

1. `1_23_XY` 有两个 mask，尺寸相同但哈希不同，前景比例也不同；需要确认哪个是最终标注，或是否分别代表不同版本。
2. `S_1` 同时存在 `S-1-Seg.nii` 和 `S_1_Seg.nii.gz`，后者标签值为 `65535` 而不是 `1`；需要统一为二值 `0/1`。
3. `nose_layer36` 的 mask 出现标签 `2`；若课程任务是二分类分割，应在预处理阶段把所有非零标签映射为 `1`，并在报告中说明。
4. 命名格式混用 `S_1`、`S-1`、`Seg(1)(1)` 等，后续扩充数据前建议统一命名。

## 建议数据规范

后续新增样本建议按以下格式放置，原始数据只读保存，训练时再生成 processed 版本：

```text
dataset/raw/<case_id>/
  image.nii.gz
  mask.nii.gz

dataset/processed/
  slices_2d/
  patches_3d/
dataset/splits/
  split_seed42.json
```

建议预处理策略：

- mask 统一转为 `uint8`，标签统一为 `0/1`。
- `.nii` 统一压缩为 `.nii.gz`；压缩不改变体素数据，只改变存储方式。
- 划分训练/验证/测试时按 case 划分，不按 slice 随机混合，避免同一体数据泄漏到训练和测试两边。

