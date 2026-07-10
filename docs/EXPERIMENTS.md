# 实验记录

每次训练后追加一行，方便最后直接整理进报告。`best Dice` 对 nnU-Net 训练中记录为验证集 pseudo Dice；最终测试集 Dice 需推理后用 `imgseg-nnunet evaluate` 计算。EfficientNet-B0 训练阶段的验证指标按 case 汇总全部 slice 的混淆计数后计算，再对 case 做宏平均；正式测试指标需由 `imgseg-efficientnet evaluate` 对保存的 checkpoint 计算。

| date | owner | model | split | input | loss | epochs | best Dice | IoU | precision | recall | infer sec/case | checkpoint | notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-07-08 | Codex | nnU-Net v2 2d fold0 | `split_seed42` | nnU-Net 2D patches | Dice+CE | 8 complete, aborted at epoch 8 | 0.7734 | 0.9301 | 0.9487 | 0.9793 | 60.5 | `checkpoints/nnunet_v2/Dataset501_ImgSeg/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth` | Preliminary test result from `checkpoint_best.pth`; training stopped due Windows shared-memory worker failure. Reports: `outputs/reports/nnunet_v2_2d_fold0`; prediction overlay: `outputs/visualizations/nnunet_v2_2d_fold0/H1-20Layer_prediction_overlay.png`. |
| 2026-07-09 | Codex | nnU-Net v2 2d fold0 200 epochs | `split_seed42` | nnU-Net 2D patches | Dice+CE | 200 complete | 0.7981 | 0.9514 | 0.9779 | 0.9723 | 58.8 | `checkpoints/nnunet_v2/Dataset501_ImgSeg/nnUNetTrainer_200epochs__nnUNetPlans__2d/fold_0/checkpoint_final.pth` | Final checkpoint selected: test Dice 0.9751 on `H1-20Layer`; `checkpoint_best.pth` test Dice 0.9751, IoU 0.9513, precision 0.9744, recall 0.9757. Reports: `outputs/reports/nnunet_v2_2d_fold0_200epochs`; final predictions: `outputs/nnunet_v2_2d_fold0_200epochs_final`; overlay: `outputs/visualizations/nnunet_v2_2d_fold0_200epochs/H1-20Layer_final_overlay.png`. |
| 2026-07-10 | NH5 | 2D U-Net + EfficientNet-B0 encoder | `split_seed42` test | z-axis 2D slices, resized to 512×512 | Dice+BCE | 100 complete | 0.7916 | 0.8313 | 0.9281 | 0.8878 | 11.2 | `checkpoints/efficientnet_b0/best.pt` (legacy) | Best validation Dice was reached at epoch 3 with ImageNet encoder initialization. Test macro Dice was 0.9065 over 2 cases on an NVIDIA GPU PC. End-to-end prediction time includes NIfTI read, inference, and mask save. Raw local records: `outputs/efficientnet_b0/log.json` and `outputs/efficientnet_b0/metrics.json`. New runs save `best.pth` and `train.log`. |


## 2026-07-10 EfficientNet-B0 正式训练与测试

在带 NVIDIA GPU 的 PC 上使用 `configs/efficientnet_b0/base.yaml` 完成 100 轮训练，并使用保存的最佳 checkpoint 在 `split_seed42` 的 test split 上正式评估。该次旧实验产物名为 `checkpoints/efficientnet_b0/best.pt`；当前代码统一使用 `checkpoints/efficientnet_b0/best.pth` 作为交付文件名。训练使用 ImageNet encoder 初始化；最佳验证 Dice 为 0.7916，出现在第 3 轮。

测试集包含 2 个 case，以下指标均为 case 宏平均；耗时为 `predict_volume` 记录的端到端时间，包含 NIfTI 读取、整卷 slice batching 推理和结果保存，不包含后续指标计算。

| case | volume shape | Dice | IoU | Precision | Recall | infer sec |
|---|---:|---:|---:|---:|---:|---:|
| `2_25_XY` | 973×973×298 | 0.8674 | 0.7658 | 0.9195 | 0.8208 | 16.25 |
| `S-4` | 512×512×331 | 0.9456 | 0.8968 | 0.9367 | 0.9547 | 6.06 |
| **macro average** | — | **0.9065** | **0.8313** | **0.9281** | **0.8878** | **11.16** |

训练 loss 继续下降时，验证 Dice 在第 3 轮后未再超过 0.7916，说明当前配置很早进入验证性能平台期，并存在过拟合可能；后续可加入 early stopping，并在固定 split 上验证数据增强或正则化调整。两个测试 case 的 Dice 相差约 0.0782，且测试集规模很小，因此当前 0.9065 的宏平均结果应视为本项目固定划分上的基线，不宜直接外推为稳定泛化性能。
