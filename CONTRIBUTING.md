# 贡献指南

感谢参与本项目。这里是一个课程性质的医学图像分割仓库，目标是用统一数据流程、统一评估指标和统一 WebUI 对比 `nnU-Net v2`、`MONAI SegResNet`、`EfficientNet-B0 encoder` 三类分割方案。

开始开发前请先阅读：

- [AGENTS.md](AGENTS.md)：项目内智能体和协作者规则。
- [docs/COLLABORATION.md](docs/COLLABORATION.md)：分工、Git 工作流和验收目标。
- [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md)：当前已知数据问题。
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)：训练与评估结果记录。

## 开发环境

本项目使用 `uv` 管理环境和命令运行：

```powershell
uv sync
uv run pytest
uv run ruff check .
```

如果本机未安装 `uv`，请先安装 `uv`，不要改用其他包管理器生成 `requirements.txt` 或提交额外锁文件。

## 分支与提交

主分支为 `main`。建议分支命名：

- `feat/data-pipeline`
- `feat/model-nnunet-v2`
- `feat/model-monai-segresnet`
- `feat/model-efficientnet-b0`
- `feat/webui`
- `fix/<short-topic>`

提交请保持小步、清晰，并避免把不相关改动揉在一起。不要提交：

- `dataset/**/*.nii`
- `dataset/**/*.nii.gz`
- `dataset/processed/`
- `checkpoints/`
- `outputs/`
- `Test_Seg/`
- `.venv/`

## 数据与实验要求

- 原始数据不要直接覆盖；预处理产物放入 `dataset/processed/` 或 `outputs/`。
- 训练、验证、测试必须按 case 划分，不能把同一个体数据的 slice 随机分散到不同集合。
- 二分类 mask 默认按 `mask > 0` 转为 `0/1`。
- 新增 NIfTI 数据脚本必须保留 affine/header，输出优先使用 `.nii.gz`。
- 未人工确认的数据问题不要擅自删除或重命名，先更新 `docs/DATASET_AUDIT.md`。

每次训练或评估后，请把关键结果追加到 `docs/EXPERIMENTS.md`，至少包含模型、split、指标、checkpoint 位置和已知问题。纯文档或 WebUI 交互改动可以说明没有新增实验。

## 代码约定

- 公共数据处理、评估指标、批量推理逻辑放在 `src/img_seg/` 下复用。
- 模型私有封装可以放在 `src/img_seg/models/<model>.py`。
- 独立可执行的小工具放在 `utils/`。
- 训练脚本必须接受配置文件路径，不要硬编码数据路径、checkpoint 路径或超参数。
- WebUI 只能调用统一推理接口，不直接依赖某个模型内部实现。

## Pull Request

发起 PR 前请确认：

- 已阅读并遵守 [AGENTS.md](AGENTS.md)。
- 已运行相关 smoke test，至少运行 `uv run pytest` 或说明无法运行的原因。
- 已运行 `uv run ruff check .`，或说明无法运行的原因。
- 涉及实验、训练或推理结果时，已更新 `docs/EXPERIMENTS.md`。
- PR 描述包含改动内容、运行方式、测试结果、数据/模型影响和已知风险。

合并到 `main` 前应通过代码评审和测试。若使用 GitHub，请通过 `gh` 或网页创建 PR，不直接在本地强推覆盖主分支。
