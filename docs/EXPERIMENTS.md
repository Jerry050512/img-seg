# 实验记录

每次训练后追加一行，方便最后直接整理进报告。`best Dice` 对 nnU-Net 训练中记录为验证集 pseudo Dice；最终测试集 Dice 需推理后用 `imgseg-nnunet evaluate` 计算。EfficientNet-B0 训练阶段的验证指标按 case 汇总全部 slice 的混淆计数后计算，再对 case 做宏平均；正式测试指标需由 `imgseg-efficientnet evaluate` 对保存的 checkpoint 计算。

| date | owner | model | split | input | loss | epochs | best Dice | IoU | precision | recall | infer sec/case | checkpoint | notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-07-08 | Codex | nnU-Net v2 2d fold0 | `split_seed42` | nnU-Net 2D patches | Dice+CE | 8 complete, aborted at epoch 8 | 0.7734 | 0.9301 | 0.9487 | 0.9793 | 60.5 | `checkpoints/nnunet_v2/Dataset501_ImgSeg/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth` | Preliminary test result from `checkpoint_best.pth`; training stopped due Windows shared-memory worker failure. Reports: `outputs/reports/nnunet_v2_2d_fold0`; prediction overlay: `outputs/visualizations/nnunet_v2_2d_fold0/H1-20Layer_prediction_overlay.png`. |
| 2026-07-09 | Codex | nnU-Net v2 2d fold0 200 epochs | `split_seed42` | nnU-Net 2D patches | Dice+CE | 200 complete | 0.7981 | 0.9514 | 0.9779 | 0.9723 | 58.8 | `checkpoints/nnunet_v2/Dataset501_ImgSeg/nnUNetTrainer_200epochs__nnUNetPlans__2d/fold_0/checkpoint_final.pth` | Final checkpoint selected: test Dice 0.9751 on `H1-20Layer`; `checkpoint_best.pth` test Dice 0.9751, IoU 0.9513, precision 0.9744, recall 0.9757. Reports: `outputs/reports/nnunet_v2_2d_fold0_200epochs`; final predictions: `outputs/nnunet_v2_2d_fold0_200epochs_final`; overlay: `outputs/visualizations/nnunet_v2_2d_fold0_200epochs/H1-20Layer_final_overlay.png`. |
| 2026-07-10 | NH5 | 2D U-Net + EfficientNet-B0 encoder | legacy `split_seed42` test snapshot | z-axis 2D slices, resized to 512×512 | Dice+BCE | 100 complete | 0.7916 | 0.8313 | 0.9281 | 0.8878 | 11.2 | `checkpoints/efficientnet_b0/20260710T152124825334Z/best.pth` | Best validation Dice was reached at epoch 3 with ImageNet encoder initialization. Test macro Dice was 0.9065 over 2 cases on an NVIDIA GPU PC. End-to-end prediction time includes NIfTI read, inference, and mask save. Raw local records: `outputs/efficientnet_b0/20260710T152124825334Z/train.log` and `metrics.json`. This is the selected EfficientNet delivery run because it outperforms the newer 30-epoch run on Dice and IoU. |
| 2026-07-10 | B | MONAI SegResNet 3D | `split_seed42` | 96x96x64 3D patches | Dice+BCE | 200 complete | 0.8261 | 0.9157 | 0.9462 | 0.9650 | 25.6 | `checkpoints/monai_segresnet/best.pt` | Best validation checkpoint selected at epoch 170. Fixed test-set mean Dice 0.9555 on `2_25_XY` and `S_3`, exceeding the 0.92 target; per-case Dice: 0.9342 and 0.9769. Metrics: `outputs/monai_segresnet/test_metrics.json`; predictions: `outputs/monai_segresnet/test_predictions`. |
| 2026-07-10 | `@Jerry050512` | nnU-Net v2 2d fold0 200 epochs (`nnunet_v2_2d_fold0_200epochs_20260710_114102`) | `split_seed42` (9/2/2 cases) | 1024×640 patches | Dice+CE | 200 complete | 0.8375 | 0.9048 | 0.9581 | 0.9405 | 427.0 | `checkpoints/nnunet_v2_runs/nnunet_v2_2d_fold0_200epochs_20260710_114102/Dataset501_ImgSeg/nnUNetTrainer_200epochs__nnUNetPlans__2d/fold_0/checkpoint_best.pth` | Selected best checkpoint: test Dice 0.9490, better than final checkpoint Dice 0.9417. Full-volume validation Dice 0.7737, IoU 0.6317, precision 0.8823, recall 0.6889. Bounded-memory preprocessing completed after standard foreground materialization hit `ArrayMemoryError`. Test used per-case, single-process, no-TTA inference; `S_3` used GPU (32.1 s), while oversized `2_25_XY` used CPU fallback (821.8 s). Visualization: `outputs/visualizations/nnunet_v2_2d_fold0_200epochs_20260710_114102`; full experiment report: `outputs/reports/nnunet_v2_2d_fold0_200epochs_20260710_114102/EXPERIMENT.md`; course report: `docs/course_report/course-report.typ`. |
| 2026-07-11 | NH5 | 2D U-Net + EfficientNet-B0 encoder | legacy `split_seed42` test snapshot | z-axis 2D slices, resized to 512×512 | Dice+BCE | 30 complete | 0.7944 | 0.8248 | 0.9332 | 0.8755 | 10.2 | `checkpoints/efficientnet_b0/20260711T031117144692Z/best.pth` | Batch size 16; best validation Dice was reached at epoch 5 with ImageNet encoder initialization. Test macro Dice was 0.9023 over the same 2 cases as the 100-epoch run. Raw local records: `outputs/efficientnet_b0/20260711T031117144692Z/train.log` and `metrics.json`. |


## 2026-07-10 EfficientNet-B0 正式训练与测试

在带 NVIDIA GPU 的 PC 上使用 `configs/efficientnet_b0/base.yaml` 完成 100 轮训练，并使用保存的最佳 checkpoint 在 `split_seed42` 的 test split 上正式评估。归档产物为 `checkpoints/efficientnet_b0/20260710T152124825334Z/best.pth`，训练日志与指标位于同 run id 的 `outputs/efficientnet_b0/` 子目录。训练使用 ImageNet encoder 初始化；最佳验证 Dice 为 0.7916，出现在第 3 轮。

测试集包含 2 个 case，以下指标均为 case 宏平均；耗时为 `predict_volume` 记录的端到端时间，包含 NIfTI 读取、整卷 slice batching 推理和结果保存，不包含后续指标计算。

| case | volume shape | Dice | IoU | Precision | Recall | infer sec |
|---|---:|---:|---:|---:|---:|---:|
| `2_25_XY` | 973×973×298 | 0.8674 | 0.7658 | 0.9195 | 0.8208 | 16.25 |
| `S-4` | 512×512×331 | 0.9456 | 0.8968 | 0.9367 | 0.9547 | 6.06 |
| **macro average** | — | **0.9065** | **0.8313** | **0.9281** | **0.8878** | **11.16** |

训练 loss 继续下降时，验证 Dice 在第 3 轮后未再超过 0.7916，说明当前配置很早进入验证性能平台期，并存在过拟合可能；后续可加入 early stopping，并在固定 split 上验证数据增强或正则化调整。两个测试 case 的 Dice 相差约 0.0782，且测试集规模很小，因此当前 0.9065 的宏平均结果应视为本项目固定划分上的基线，不宜直接外推为稳定泛化性能。

## 2026-07-11 nnU-Net 交付 checkpoint 精简验证

原始 `checkpoint_best.pth` 为 370,794,358 bytes（353.62 MiB），其中网络 FP32 权重和 optimizer state 各占约 176.71 MiB。使用 `utils/export_nnunet_checkpoint.py` 删除仅用于断点续训的 optimizer、grad scaler 和日志字段后，得到 nnU-Net 原生推理接口可读取的 `checkpoint_best_inference.pth`：185,411,378 bytes（176.82 MiB），缩小 50.0%，低于课程 200 MB 单文件上限。导出过程不改变 dtype，并逐张量验证权重完全一致；文件 SHA-256 为 `6abfe3194dc01a9cbbf2f473a7a5ba9fbf290a98e5440fc8bd88e51e8f0375f5`。

通过根目录 `test.py` 在 RTX 4060 Laptop 上对 `S_3` 重新推理，得到 Dice 0.9831、Precision 0.9772、Recall 0.9891、PixelAcc 0.9855，与原 checkpoint 的既有同机结果一致。该交付 checkpoint 仅可用于推理，不能用于恢复训练。
