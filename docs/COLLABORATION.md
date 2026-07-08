# 三人协作规范

项目固定对比三个模型：`nnU-Net v2`、`MONAI SegResNet`、`EfficientNet-B0 encoder` 的 2D 轻量分割模型。所有模型必须共享数据划分、评估指标和批量推理接口。

## 角色分工

| 角色 | 主要任务 | 交付 |
|---|---|---|
| A：nnU-Net v2 负责人 | 数据转换到 nnU-Net 格式、训练 2D/3D lowres、记录最佳 Dice | `configs/nnunet_v2/`、训练日志、权重路径、推理适配器 |
| B：MONAI SegResNet 负责人 | 基于 MONAI 训练 3D SegResNet，调 patch size、AMP、batch size | `configs/monai_segresnet/`、训练配置、权重、指标 |
| C：EfficientNet-B0/WebUI 负责人 | 训练 2D EfficientNet-B0 encoder 轻量模型，统一推理接口，做 WebUI 批量预测 | `configs/efficientnet_b0/`、WebUI、导出 `Test_Seg` |

公共部分由所有人共同遵守：

- 数据读取、预处理、划分只保留一套实现。
- 指标计算只保留一套实现，至少包含 Dice、IoU、Precision、Recall、推理耗时。
- 每个模型都通过统一接口暴露 `load()` 与 `predict_volume()`，WebUI 不直接关心模型内部实现。
- 环境统一使用 `uv`，依赖统一写入 `pyproject.toml`。

## Git 工作流

仓库主分支为 `main`。分支命名：

- `feat/data-pipeline`
- `feat/model-nnunet-v2`
- `feat/model-monai-segresnet`
- `feat/model-efficientnet-b0`
- `feat/webui`
- `fix/<short-topic>`

合并规则：

- 每个功能分支合并前至少跑一次 smoke test。
- 不提交 `dataset/**/*.nii*`、`checkpoints/`、`outputs/`、`.venv/`。
- 训练结果用表格记录到 `docs/EXPERIMENTS.md`，权重文件通过网盘/共享盘传递。
- PR 或合并说明必须包含：改了什么、怎么运行、当前指标、已知问题。
- 智能体或成员开发前必须读取根目录 [AGENTS.md](../AGENTS.md)。

## 统一接口草案

后续模型封装建议统一成：

```python
class Segmenter:
    name: str

    def load(self, checkpoint_path: str) -> None:
        ...

    def predict_volume(self, image_path: str, output_path: str) -> dict:
        ...
```

`predict_volume()` 返回：

```python
{
    "case_id": "S-3",
    "output_path": ".../S-3_Seg.nii.gz",
    "elapsed_sec": 1.23,
    "shape": [512, 512, 291]
}
```

## 验收目标对齐

课程要求可以选择文件夹，对其中图片/样本批量预测，并输出二值分割结果到 `Test_Seg`。项目内建议同时支持：

- NIfTI：`*.nii`、`*.nii.gz`，输出 NIfTI mask。
- 2D 图片：`*.jpg`、`*.png`，输出同名二值 `png`，满足验收时的 `0001.jpg` 场景。
