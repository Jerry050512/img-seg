# 实验记录

每次训练后追加一行，方便最后直接整理进报告。`best Dice` 对 nnU-Net 训练中记录为验证集 pseudo Dice；最终测试集 Dice 需推理后用 `imgseg-nnunet evaluate` 计算。

| date | owner | model | split | input | loss | epochs | best Dice | IoU | precision | recall | infer sec/case | checkpoint | notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-07-08 | Codex | nnU-Net v2 2d fold0 | `split_seed42` | nnU-Net 2D patches | Dice+CE | 8 complete, aborted at epoch 8 | 0.7734 | 0.9301 | 0.9487 | 0.9793 | 60.5 | `checkpoints/nnunet_v2/Dataset501_ImgSeg/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth` | Preliminary test result from `checkpoint_best.pth`; training stopped due Windows shared-memory worker failure. Reports: `outputs/reports/nnunet_v2_2d_fold0`; prediction overlay: `outputs/visualizations/nnunet_v2_2d_fold0/H1-20Layer_prediction_overlay.png`. |
