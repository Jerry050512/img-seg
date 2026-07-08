# 项目结构

项目采用 `uv + src layout`。数据、模型配置、训练产物和报告素材分离，方便三个人并行开发。

```text
img-seg/
  AGENTS.md
  pyproject.toml
  README.md
  .gitignore
  configs/
    data/
      split_seed42.yaml
    nnunet_v2/
      base.yaml
    monai_segresnet/
      base.yaml
    efficientnet_b0/
      base.yaml
  dataset/
    <case_id>/
      *.nii
      *.nii.gz
  docs/
    COLLABORATION.md
    DATASET_AUDIT.md
    EXPERIMENTS.md
    MODEL_RESEARCH.md
    PROJECT_STRUCTURE.md
  src/img_seg/
    data/
    evaluation/
    inference/
    models/
    training/
    webui/
  tests/
  utils/
```

## 模型边界

三条模型线固定如下：

| model key | model | owner scope | config |
|---|---|---|---|
| `nnunet_v2` | nnU-Net v2 | 数据转换、nnU-Net 训练/推理命令封装 | `configs/nnunet_v2/base.yaml` |
| `monai_segresnet` | MONAI SegResNet | 3D patch 训练、AMP、Dice+BCE/Tversky | `configs/monai_segresnet/base.yaml` |
| `efficientnet_b0` | EfficientNet-B0 encoder | 2D 切片、轻量模型、WebUI 默认快速推理 | `configs/efficientnet_b0/base.yaml` |

## 公共模块边界

- `src/img_seg/data/`：case 扫描、NIfTI 读取、mask 二值化、split、2D slice/3D patch 生成。
- `src/img_seg/evaluation/`：Dice、IoU、Precision、Recall、HD95、耗时统计。
- `src/img_seg/inference/`：统一批量推理入口，负责目录输入、输出命名、保存 mask。
- `src/img_seg/models/`：三个模型的适配器，向外暴露统一 Segmenter 接口。
- `src/img_seg/training/`：训练入口和配置加载。
- `src/img_seg/webui/`：WebUI，只调用 `inference` 层。

## 数据与产物

- `dataset/`：本地原始数据，不进 Git。
- `dataset/processed/`：预处理产物，不进 Git。
- `dataset/splits/`：可复现划分文件，可提交小型 `.json/.yaml`，不提交大体数据。
- `checkpoints/`：模型权重，不进 Git。
- `outputs/`：预测结果、图表、日志，不进 Git。
