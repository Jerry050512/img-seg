# AI Image Segmentation Course Project

本项目用于人工智能综合课程实践：基于 NIfTI 扫描样本及其 segmentation mask，训练并对比 3 个分割模型，最终提供可选择模型的批量推理 WebUI。

> 班级 / 组长姓名：提交前由小组补充（仓库现有材料仅记录账号 `@Jerry050512`、`@NH-5`、`@flypigff`）。

## 快速验收

当前代码要求 **Python 3.12**。项目正式使用 `uv` 与 `uv.lock`；课程要求的
`requirements.txt` 仅作为 deprecated 的 `pip` 兼容入口，依赖真值仍是
`pyproject.toml`。以下命令均可从仓库根目录直接复制。

环境配置（推荐）：

```bash
uv sync
```

环境配置（课程兼容/deprecated）：

```bash
python -m pip install -r requirements.txt
```

批量测试: `test.py` 的 stdout 最后一行始终是标准 JSON，并包含 `mIoU`、
`Dice`、`precision`、`recall`、`PixelAcc`、`FPS` 与 `Params(M)`：

```bash
uv run python test.py --model efficientnet_b0 --data-path /data/evaluation/test/ --weight-path ./best_model.pth --output-dir ./outputs/course_test
```

如果图像与 Ground Truth 分别位于不同目录：

```bash
uv run python test.py --model efficientnet_b0 --data-path /data/evaluation/test/images --reference-path /data/evaluation/test/labels --weight-path ./best_model.pth
```

单个 NIfTI 推理并保存二值 mask 与可视化预览：

```bash
uv run python predict.py --model efficientnet_b0 --image /data/evaluation/sample.nii.gz --weight ./best_model.pth --output ./result.png
```

文件夹批量推理（支持 50 个及更多 NIfTI；mask 写入 `Test_Seg/`，预览写入
`Test_Seg/previews/`）：

```bash
uv run python predict.py --model efficientnet_b0 --input-dir /data/evaluation/test/ --weight ./best_model.pth --output-dir ./Test_Seg
```

训练模型（目录格式为 `case_id/image.nii.gz` 与
`case_id/mask.nii.gz`）：

```bash
uv run python train.py --model efficientnet_b0 --config configs/efficientnet_b0/base.yaml --data-path /path/to/train_dataset
```

选择另外两个模型时，把 `--model` 改为 `monai_segresnet` 或 `nnunet_v2`，并传入
对应配置与 checkpoint。SegResNet 只接受 3D NIfTI；nnU-Net 的 `train.py` 默认依次
执行数据转换、planning/preprocessing 和 fold 0 训练，可用 `--skip-prepare` 或
`--skip-plan` 复用已有产物。课程说明中的无前缀写法也兼容，例如：

```bash
uv run python predict.py image /data/evaluation/sample.nii.gz weight ./best_model.pth output ./result.png
```

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
- EfficientNet-B0 工作流：[docs/EFFICIENTNET_B0_WORKFLOW.md](docs/EFFICIENTNET_B0_WORKFLOW.md)
- MONAI SegResNet 工作流：[docs/MONAI_SEGRESNET_WORKFLOW.md](docs/MONAI_SEGRESNET_WORKFLOW.md)
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
uv run imgseg-segresnet --config configs/monai_segresnet/base.yaml inspect
uv run python utils/compress_nii.py dataset --recursive
uv run imgseg-efficientnet train --config configs/efficientnet_b0/base.yaml
uv run imgseg-predict --model efficientnet_b0 --checkpoint checkpoints/efficientnet_b0/20260710T083015123456Z/best.pth --input-dir Test --output-dir Test_Seg
```

## WebUI

```powershell
uv run imgseg-webui
```

WebUI 通过公共批量推理接口提供 `nnU-Net v2`、`MONAI SegResNet 3D` 与 `EfficientNet-B0 2D U-Net` 模型选择，并按所选模型列出可用 checkpoint；也可以手动填写权重路径。输入支持单个文件或文件夹中的 NIfTI，以及 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff` 图片；SegResNet 仅支持 3D NIfTI。默认输出目录为 `Test_Seg`，NIfTI 输出 `.nii.gz` 二值 mask，2D 图片输出同名 `.png` 二值 mask。

普通图片只表达像素空间分割，不包含真实物理间距。选择 nnU-Net v2 时，图片会先按单通道单 slice 转为使用 identity affine 的临时 NIfTI；选择 EfficientNet-B0 时，图片按灰度 2D 输入直接预测。

## 社区与许可

欢迎通过 issue 和 pull request 参与改进。提交前请阅读 [贡献指南](CONTRIBUTING.md)，并确认没有提交原始医学影像、训练权重或预测结果。

本项目代码使用 [MIT License](LICENSE)。数据集、模型权重和课程材料如有单独授权，应以其原始授权和课程要求为准。
