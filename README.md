# AI Image Segmentation Course Project

本项目用于人工智能综合课程实践：基于 NIfTI 扫描样本及其 segmentation mask，训练并对比 3 个分割模型，最终提供可选择模型的批量推理 WebUI。

## 当前状态

- 原始数据位于 `dataset/`，读取布局规则位于 `configs/data/dataset.yaml`。
- 原始样本目录已规范为 `dataset/<case_id>/image.nii.gz` 与 `mask.nii.gz`。
- 项目环境使用 `uv` 管理，依赖声明在 `pyproject.toml`。
- 固定对比三个模型：`nnU-Net v2`、`MONAI SegResNet`、`EfficientNet-B0 encoder` 的 2D 轻量分割模型。

## 目录结构

```text
dataset/                 # 本地数据，不建议直接提交到 Git
configs/                 # 三个模型与数据流程配置
docs/                    # 数据审计、模型调研、协作规范、报告素材
src/img_seg/             # 项目源码
  data/                  # NIfTI 读取、切片、增强、划分
  models/                # 各模型封装与统一接口
  training/              # 训练入口
  inference/             # 单模型/批量推理
  evaluation/            # Dice、IoU、HD95 等评估
  webui/                 # Gradio WebUI
utils/                   # 独立工具脚本
tests/                   # 单元测试与 smoke test
checkpoints/             # 训练权重，本地生成，不提交
outputs/                 # 预测、指标、图表，本地生成，不提交
```

更详细的结构约定见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 环境

本机如果还没有安装 `uv`，先安装 uv 后再同步依赖。Windows PowerShell 推荐：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
```

如果需要明确安装 CUDA 版 PyTorch，可先按 PyTorch 官方命令用 `uv pip install` 安装对应 CUDA wheel，再执行 `uv sync` 校验其余依赖。

## 文档入口

- 数据集盘点：[docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md)
- 项目结构：[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- nnU-Net v2 工作流：[docs/NNUNET_V2_WORKFLOW.md](docs/NNUNET_V2_WORKFLOW.md)
- 模型调研与推荐：[docs/MODEL_RESEARCH.md](docs/MODEL_RESEARCH.md)
- 三人协作规范：[docs/COLLABORATION.md](docs/COLLABORATION.md)
- 贡献指南：[CONTRIBUTING.md](CONTRIBUTING.md)
- 行为准则：[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- 安全政策：[SECURITY.md](SECURITY.md)
- 智能体开发规则：[AGENTS.md](AGENTS.md)

## 常用工具

```powershell
uv run python utils/audit_nifti.py dataset --labels
uv run imgseg-nnunet --config configs/nnunet_v2/base.yaml prepare
uv run python utils/compress_nii.py dataset --recursive
```

## WebUI

```powershell
uv run imgseg-webui
```

WebUI 支持选择 nnU-Net v2 权重，对单个 NIfTI、NIfTI 文件夹、单张 `.jpg/.png` 或图片文件夹进行批量推理。默认输出目录为 `Test_Seg`，NIfTI 输出 `.nii.gz` 二值 mask，2D 图片输出同名 `.png` 二值 mask。
普通图片会先按单通道单 slice 转为临时 NIfTI，并使用 identity affine，因此结果只表达像素空间分割，不包含真实物理间距。

## 社区与许可

欢迎通过 issue 和 pull request 参与改进。提交前请阅读 [贡献指南](CONTRIBUTING.md)，并确认没有提交原始医学影像、训练权重或预测结果。

本项目代码使用 [MIT License](LICENSE)。数据集、模型权重和课程材料如有单独授权，应以其原始授权和课程要求为准。
