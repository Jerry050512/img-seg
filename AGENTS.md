# AGENTS.md

本文件是项目内所有智能体和协作者的工作规则。任何自动化开发、代码生成、重构、训练脚本改动、文档补充，都必须优先遵守这里的约定。

## 环境规则

- 使用 `uv` 管理环境和命令运行。
- 新依赖必须写入 `pyproject.toml`，不要新增或恢复 `requirements.txt`。
- 常用命令：

```powershell
uv sync
uv run python utils/audit_nifti.py dataset --labels
uv run pytest
uv run ruff check .
```

- 如果本机未安装 `uv`，只更新项目文件和文档，不要用其他包管理器偷偷安装依赖。最好向用户发起请求并自行安装 `uv`。
- 不要把虚拟环境、缓存、训练权重或预测结果提交到 Git。

## 数据规则

- `dataset/` 视为本地原始数据目录，不提交 `.nii` 或 `.nii.gz` 文件。
- 原始数据不要直接覆盖。预处理产物放入 `dataset/processed/` 或 `outputs/`。
- 训练/验证/测试必须按 case 划分，不能把同一个体数据的 slice 随机分散到不同集合。
- 二分类任务中 mask 统一按 `mask > 0` 转为 `0/1`，除非任务文档明确要求多类别。
- 新增数据脚本必须保留 affine/header，输出优先使用 `.nii.gz`。
- 当前已知需要人工确认的数据问题记录在 `docs/DATASET_AUDIT.md`，不要在未确认时删除重复 mask。

## 协作规则

- 三个模型的训练配置分别放入对应 `configs/<model>/`。
- 模型私有代码可以在 `src/img_seg/models/<model>.py`，但数据处理、指标、批量推理必须复用公共模块。
- 独立可执行的小工具放 `utils/`；可被项目导入的库代码放 `src/img_seg/`。
- 合并前必须更新 `docs/EXPERIMENTS.md` 或说明为什么本次没有实验结果。
- 合并前至少运行与改动相关的 smoke test；不能运行时要在提交说明中写明原因。

## Git 规则

- 主分支使用 `main`。
- 分支命名：
  - `feat/data-pipeline`
  - `feat/model-nnunet-v2`
  - `feat/webui`
  - `fix/<short-topic>`
- 不要提交：
  - `dataset/**/*.nii`
  - `dataset/**/*.nii.gz`
  - `checkpoints/`
  - `outputs/`
  - `.venv/`
- 不要重写、删除或移动他人未说明的改动。遇到冲突时先读上下文，再做最小修改。

## 代码规则

- 优先复用项目已有接口和配置，不要每个模型各写一套数据读取、指标计算和推理流程。
- 公共评估指标至少包含 Dice、IoU、Precision、Recall、推理耗时。
- 训练脚本必须接受配置文件路径，不要把路径和超参数硬编码在代码里。
- WebUI 只能调用统一推理接口，不直接依赖某个模型内部实现。
- 代码保持小步提交，文档和配置随代码同步更新。

### 合并入 main 分支

- 请使用 `gh` 命令创建 PR 并等待合并审核。
- 如未安装，请提示用户安装。
- 合并前，务必对代码更改进行 code review，并通过测试。

## 交付规则

每个模型负责人最终至少交付：

- 训练配置。
- 训练/推理命令。
- 最佳 checkpoint 的外部存储位置或命名。
- 验证/测试指标。
- 简短实验结论，写入 `docs/EXPERIMENTS.md` 或报告素材文档。

WebUI 最终至少支持：

- 模型选择。
- 输入目录选择或填写。
- 输出目录选择或默认 `Test_Seg`。
- 批量推理进度显示。
- 输出二值 mask。
