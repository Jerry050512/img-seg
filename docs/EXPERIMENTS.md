# 实验记录

每次训练后追加一行，方便最后直接整理进报告。`best Dice` 对 nnU-Net 训练中记录为验证集 pseudo Dice；最终测试集 Dice 需推理后用 `imgseg-nnunet evaluate` 计算。EfficientNet-B0 训练阶段的验证指标按 case 汇总全部 slice 的混淆计数后计算，再对 case 做宏平均；正式测试指标需由 `imgseg-efficientnet evaluate` 对保存的 checkpoint 计算。

| date | owner | model | split | input | loss | epochs | best Dice | IoU | precision | recall | infer sec/case | checkpoint | notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-07-08 | Codex | nnU-Net v2 2d fold0 | `split_seed42` | nnU-Net 2D patches | Dice+CE | 8 complete, aborted at epoch 8 | 0.7734 | 0.9301 | 0.9487 | 0.9793 | 60.5 | `checkpoints/nnunet_v2/Dataset501_ImgSeg/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth` | Preliminary test result from `checkpoint_best.pth`; training stopped due Windows shared-memory worker failure. Reports: `outputs/reports/nnunet_v2_2d_fold0`; prediction overlay: `outputs/visualizations/nnunet_v2_2d_fold0/H1-20Layer_prediction_overlay.png`. |
| 2026-07-09 | Codex | nnU-Net v2 2d fold0 200 epochs | `split_seed42` | nnU-Net 2D patches | Dice+CE | 200 complete | 0.7981 | 0.9514 | 0.9779 | 0.9723 | 58.8 | `checkpoints/nnunet_v2/Dataset501_ImgSeg/nnUNetTrainer_200epochs__nnUNetPlans__2d/fold_0/checkpoint_final.pth` | Final checkpoint selected: test Dice 0.9751 on `H1-20Layer`; `checkpoint_best.pth` test Dice 0.9751, IoU 0.9513, precision 0.9744, recall 0.9757. Reports: `outputs/reports/nnunet_v2_2d_fold0_200epochs`; final predictions: `outputs/nnunet_v2_2d_fold0_200epochs_final`; overlay: `outputs/visualizations/nnunet_v2_2d_fold0_200epochs/H1-20Layer_final_overlay.png`. |

WebUI-only changes do not add experiment rows unless they run a new model evaluation.

## 2026-07-10 EfficientNet-B0 工程优化说明

本轮接入共享批量推理与 WebUI 模型选择，并实现整卷 slice batching、CUDA 推理 AMP、验证集全 slice 的 case-level 聚合，以及有界 NIfTI proxy cache；同时补齐 `--config` 参数顺序兼容和 BMP/TIFF 输入支持。

本轮没有运行新的 GPU 训练或正式模型评估，没有产生可交付的 EfficientNet-B0 checkpoint，也没有新增可报告的 Dice、IoU、Precision、Recall 或推理耗时。上述改动不构成精度提升或加速结论，因此不在实验表中添加占位结果。取得外部提供的兼容 checkpoint 后，应在固定 split 和硬件上重新评估，并把权重位置、配置、指标与耗时追加到上表。
