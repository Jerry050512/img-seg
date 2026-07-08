# 分割模型调研与推荐

目标硬件以 RTX 4060 8GB 为基准。当前样本数量少、体数据分辨率差异较大，因此推荐先保证可复现实验链路，再逐步加复杂模型。

## 推荐三模型组合

| 模型 | 定位 | 推荐原因 | 4060 8GB 策略 | 风险 |
|---|---|---|---|---|
| nnU-Net v2 | 强基线 / 接近 SOTA | 自动分析数据、配置 2D/3D U-Net，适合医学/生物医学 NIfTI 分割；可作为报告中最有说服力的主模型 | 先跑 2D 或 3D lowres/fullres 小 patch，batch size 视显存调低 | 框架目录格式要求严格，初次配置稍繁琐 |
| MONAI SegResNet | 可控 3D CNN | 代码自主性更强，便于报告解释网络结构、训练循环和损失函数 | 3D patch 训练，例如 96x96x64 或更小；启用 AMP | 数据太少时容易过拟合，需要增强 |
| 2D U-Net + EfficientNet-B0 encoder | 轻量化模型 | 训练和推理快，适合作为 WebUI 默认快速模型，也能体现效率对比 | 将体数据切片为 2D；输入 512；batch size 4-8 视显存 | 缺少 3D 上下文，边界连续性可能弱 |

## 备选/加分模型

Swin UNETR 可以作为加分尝试：它属于 Transformer + U-Net 风格的 3D 医学分割模型，长程建模能力强，但对数据量和显存更敏感。以当前 8GB 显存与小样本阶段，不建议作为必须交付模型；若时间充裕，可用很小 patch 和 AMP 做对照实验。

SAM/MedSAM 类模型更适合交互式或提示驱动分割。课程验收要求是选择文件夹批量推理输出二值图，除非后续能稳定自动生成 prompt，否则不建议作为三大主模型之一。

## 评估指标

主表建议包含：

- Dice coefficient：核心精度指标，对应课程 92% 目标。
- IoU/Jaccard：与 Dice 互补，便于报告分析。
- Precision、Recall：观察漏分和误分。
- Inference time / volume：体现效率。
- Params、显存峰值：体现轻量化。
- HD95：如果边界质量重要，可作为附加指标。

## 训练建议

- 先做 case-level split，避免按 slice 随机切分造成数据泄漏。
- 数据少时优先 5-fold 或留一法验证；报告中说明当前数据是初版，后续会扩充。
- 损失函数可用 `DiceLoss + BCEWithLogitsLoss`；类别极不平衡时可加入 Focal/Tversky。
- 2D 模型切片时过滤全空 mask 的比例，避免训练集被背景主导。
- 所有 mask 预处理统一执行 `mask = mask > 0`。

## 参考来源

- nnU-Net 官方仓库说明其会自动分析训练数据、生成数据 fingerprint、配置 U-Net 变体，并提供从预处理、训练、模型选择到推理的端到端流程：https://github.com/MIC-DKFZ/nnUNet
- nnU-Net Nature Methods 论文报告其为医学图像分割的 self-configuring 方法：https://www.nature.com/articles/s41592-020-01008-z
- Swin UNETR 论文/项目说明其使用 Swin Transformer 编码器与 CNN 解码器进行 3D 医学分割：https://arxiv.org/abs/2201.01266
- segmentation-models-pytorch 文档列出 U-Net、U-Net++、DeepLabV3、DeepLabV3+、SegFormer 等架构，并支持多种 encoder：https://smp.readthedocs.io/en/latest/


