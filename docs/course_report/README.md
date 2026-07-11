# 课程实验报告

本目录使用本地定制的 **Atlas Academic** Typst 学术技术报告模板，适合中文课程报告：A4 版式、封面、目录、章节层级、统一表格、公式、图注、页眉页脚和参考文献均已配置。

## 文件

- `course-report.typ`：报告正文。
- `report-template.typ`：可复用的 Typst 版式模板。
- `references.bib`：报告引用文献。
- `generate_assets.py`：从 nnU-Net 与 SegResNet 本地运行产物生成收敛曲线、三模型雷达图、Bad Case 图和复杂度统计。
- `assets/`：可提交的报告图表与计算口径 JSON。
- `output/pdf/img-seg-course-report.pdf`：仓库根目录下的最终 PDF。

## 构建

在仓库根目录运行：

```powershell
uv run python docs/course_report/generate_assets.py `
  --run-id nnunet_v2_2d_fold0_200epochs_20260710_114102
New-Item -ItemType Directory -Force output/pdf | Out-Null
typst compile docs/course_report/course-report.typ output/pdf/img-seg-course-report.pdf
```

Bash 用户可用 `mkdir -p output/pdf` 创建输出目录。Typst 通常会自动创建父目录，但构建说明显式创建它，便于在不同版本和 CI 环境中复现。`--run-id` 默认为报告当前使用的运行；指定其他运行时，脚本会检查所需 CSV、metrics、predictions 和 plans，缺失时列出明确路径。

重新生成 assets 需要本机保留以下被 Git 忽略的实验产物：

- `outputs/reports/nnunet_v2_2d_fold0_200epochs_20260710_114102/`
- `outputs/nnunet_v2_2d_fold0_200epochs_20260710_114102_best/`
- `checkpoints/nnunet_v2_runs/nnunet_v2_2d_fold0_200epochs_20260710_114102/`
- `dataset/processed/nnunet_raw/Dataset501_ImgSeg/`
- `outputs/monai_segresnet/history.json`
- `outputs/monai_segresnet/test_predictions/`
- `dataset/<case_id>/image.nii.gz` 与 `mask.nii.gz`

若只修改文字或表格，可直接运行 Typst 编译，不需要重新生成 assets。

## 字体

- 英文正文使用 `Libertinus Serif`。
- 中文正文保留项目原定回退路线：`LXGW WenKai -> Songti SC -> Source Han Serif -> SimSun -> Microsoft YaHei`，并在末尾追加 Linux 常见的 `Noto Serif CJK SC/JP`。
- 标题/图表采用 `LXGW WenKai -> Microsoft YaHei -> Noto Sans CJK SC/JP -> DejaVu Sans`；代码字体采用 `Consolas -> DejaVu Sans Mono`，英文正文使用 `Libertinus Serif`。

Python 绘图脚本按同一跨平台候选顺序选择本机实际存在的字体，避免 matplotlib 对缺失 family 反复告警。安装或更新字体后，建议用 `typst fonts` 确认字体，再重新编译报告。

## 提交前占位项

- 三位成员的真实姓名与学号、最终工作量比例。
- 三个最佳 checkpoint 的外部存储链接。
- 将 SegResNet 注册到统一 batch inference 与 WebUI 模型下拉框。
- 如需严格横向排名，在同一测试 case 与同一 GPU 上重新评估 EfficientNet 和三模型速度。

三模型现有实验结果已写入正文。EfficientNet 的 mIoU/Accuracy 为由公开的四舍五入逐例指标与数据审计前景比例近似恢复的值，测试第二例也与另外两模型不同；正文和 `assets/model_comparison.json` 均明确记录该限制。

EfficientNet 原始 Dice/IoU/Precision/Recall/耗时来自不可变提交 `57eec20` 的 `docs/EXPERIMENTS.md` 与 `docs/EFFICIENTNET_B0_WORKFLOW.md`。来源文档提及的本地 `log.json`、`metrics.json` 未进入 Git；`model_comparison.json` 保存了该 provenance 和推导说明。

此外，已完成运行的 validation 成员也因分支演进而不同：nnU-Net 使用 `nose_layer12`，SegResNet 使用 `nose_layer4`。报告只把各自 validation 用于 checkpoint 选择，不横向比较验证分数。
