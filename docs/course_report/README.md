# 课程实验报告

本目录使用本地定制的 **Atlas Academic** Typst 学术技术报告模板，适合中文课程报告：A4 版式、封面、目录、章节层级、统一表格、公式、图注、页眉页脚和参考文献均已配置。

## 文件

- `course-report.typ`：报告正文。
- `report-template.typ`：可复用的 Typst 版式模板。
- `references.bib`：报告引用文献。
- `generate_assets.py`：从 2026-07-10 最终 nnU-Net 运行产物生成曲线、雷达图、Bad Case 图和复杂度统计。
- `assets/`：可提交的报告图表与计算口径 JSON。
- `output/pdf/img-seg-course-report.pdf`：仓库根目录下的最终 PDF。

## 构建

在仓库根目录运行：

```powershell
uv run python docs/course_report/generate_assets.py
typst compile docs/course_report/course-report.typ output/pdf/img-seg-course-report.pdf
```

重新生成 assets 需要本机保留以下被 Git 忽略的实验产物：

- `outputs/reports/nnunet_v2_2d_fold0_200epochs_20260710_114102/`
- `outputs/nnunet_v2_2d_fold0_200epochs_20260710_114102_best/`
- `checkpoints/nnunet_v2_runs/nnunet_v2_2d_fold0_200epochs_20260710_114102/`
- `dataset/processed/nnunet_raw/Dataset501_ImgSeg/`

若只修改文字或表格，可直接运行 Typst 编译，不需要重新生成 assets。

## 字体

- 英文正文使用 `Libertinus Serif`。
- 中文正文优先使用 `LXGW WenKai`。可从 [LXGW WenKai 最新版本](https://github.com/lxgw/LxgwWenKai/releases/latest) 下载并安装。
- 中文字体回退路线为：`LXGW WenKai -> Songti SC -> Source Han Serif -> SimSun -> Microsoft YaHei`。

Typst 会按以上顺序寻找可覆盖对应字符的字体。安装或更新字体后，建议重新启动终端并用 `typst fonts` 确认字体已经被识别，再重新编译报告。
若 `Songti SC` 或 `Source Han Serif` 未安装，Typst 可能给出 unknown font family 警告；只要前序的 `LXGW WenKai` 可用，正文仍会正常排版，否则会继续回退到 `SimSun` 和 `Microsoft YaHei`。

## 提交前占位项

- 三位成员的真实姓名与学号、最终工作量比例。
- MONAI SegResNet 的 checkpoint、指标、速度、复杂度、曲线和定性图。
- EfficientNet-B0 U-Net 的 checkpoint、指标、速度、复杂度、曲线和定性图。
- nnU-Net 最佳 checkpoint 的外部存储链接。

报告不会用估计值替代这些尚未完成的实验结果；所有占位项在正文中以红色标出。
