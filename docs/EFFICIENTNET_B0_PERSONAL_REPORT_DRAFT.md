# 医学图像二值分割实验报告

## 个人负责部分草稿：2D U-Net + EfficientNet-B0 Encoder

> 文档状态：训练进行中，待补充项统一使用 `【待补充】` 标记。
> 本文只撰写本人负责的 EfficientNet-B0 模型、训练评估与批量推理部分；nnU-Net v2、MONAI SegResNet 及三模型最终横向对比由小组汇总时补入。

---

## 报告封面

| 项目 | 内容 |
|---|---|
| 课程名称 | 【待补充】 |
| 实验题目 | 基于深度学习的医学图像二值分割 |
| 小组名称/编号 | 【待补充】 |
| 姓名 | 【待补充】 |
| 学号 | 【待补充】 |
| 专业班级 | 【待补充】 |
| 指导教师 | 【待补充】 |
| 个人负责模型 | 2D U-Net + EfficientNet-B0 Encoder |
| 报告日期 | 【待补充】 |

---

## 目录

1. [实验目标](#1-实验目标)
2. [算法基本原理](#2-算法基本原理)
3. [深度学习模型基本架构与模型定位](#3-深度学习模型基本架构与模型定位)
4. [EfficientNet-B0 分割网络的层结构与参数设置](#4-efficientnet-b0-分割网络的层结构与参数设置)
5. [数据处理流程](#5-数据处理流程)
6. [网络训练、验证与推理过程](#6-网络训练验证与推理过程)
7. [实验结果与对比分析](#7-实验结果与对比分析)
8. [个人完成的工作与可运行代码](#8-个人完成的工作与可运行代码)
9. [个人对深度学习的理解和练习](#9-个人对深度学习的理解和练习)
10. [实验总结](#10-实验总结)
11. [参考文献](#11-参考文献)
12. [待补充数据清单](#12-待补充数据清单)

---

# 正文

## 1. 实验目标

本实验面向三维 NIfTI 医学影像的二值分割任务。输入为一个病例的灰度体数据，输出为与原始体数据空间尺寸一致的二值掩膜，其中背景为 0、目标区域为 1。本人负责构建以 EfficientNet-B0 为编码器的轻量级 2D U-Net 分割模型，并完成从数据切片、训练验证、权重保存到批量推理的完整流程。

具体目标如下：

1. 将三维 NIfTI 体数据按病例进行训练集、验证集和测试集划分，避免同一病例的切片进入不同集合而造成数据泄漏。
2. 将三维体数据沿 z 轴转换为二维切片，建立适用于轻量级 2D 模型的数据读取与预处理流程。
3. 使用 EfficientNet-B0 提取多尺度特征，并通过 U-Net 解码器和跳跃连接恢复空间细节，完成像素级二分类。
4. 使用 Dice 与 BCE 的组合损失缓解前景与背景不平衡问题，并使用 Dice、IoU、Precision、Recall 评价模型。
5. 保存可复现的训练日志和 `.pth` checkpoint，为后续测试、模型对比与界面调用提供依据。
6. 支持对 NIfTI 体数据及常见二维图片进行批量推理，并输出二值分割结果。

## 2. 算法基本原理

### 2.1 二值语义分割

语义分割要求模型为输入图像中的每个像素预测类别。本任务只有目标与背景两类，因此模型最后输出单通道 logits：

\[
z_{ij} = f_\theta(x)_{ij},
\]

其中，\(x\) 为输入切片，\(f_\theta\) 为参数为 \(\theta\) 的分割网络，\(z_{ij}\) 为像素 \((i,j)\) 的输出。经过 Sigmoid 函数得到前景概率：

\[
p_{ij}=\sigma(z_{ij})=\frac{1}{1+e^{-z_{ij}}}.
\]

推理时采用阈值 \(t=0.5\) 得到二值预测：

\[
\hat y_{ij}=\mathbb{I}(p_{ij}\ge t).
\]

### 2.2 Dice 与 BCE 组合损失

训练采用 Binary Cross Entropy 与软 Dice 损失之和：

\[
\mathcal L = \mathcal L_{BCE} + \mathcal L_{Dice}.
\]

BCE 逐像素约束预测概率与真实标签：

\[
\mathcal L_{BCE}
=-\frac{1}{N}\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
\]

当前实现对每个样本计算带平滑项的软 Dice：

\[
Dice_{soft}
=\frac{2\sum_i p_i y_i+1}{\sum_i p_i+\sum_i y_i+1},
\qquad
\mathcal L_{Dice}=1-\operatorname{mean}(Dice_{soft}).
\]

BCE 提供稳定的逐像素监督，Dice 则直接关注预测区域与真实区域的重叠程度。二者组合比单独使用像素准确率更适合前景占比较小的医学分割任务。

### 2.3 评估指标

设 TP、FP、FN、TN 分别表示真正例、假正例、假负例和真负例，则本实验使用：

\[
Dice=\frac{2TP}{2TP+FP+FN},
\]

\[
IoU=\frac{TP}{TP+FP+FN},
\]

\[
Precision=\frac{TP}{TP+FP},
\qquad
Recall=\frac{TP}{TP+FN}.
\]

Dice 与 IoU 衡量区域重叠程度；Precision 反映误分情况；Recall 反映漏分情况。除精度指标外，还记录每个病例的端到端推理时间，以分析轻量级模型的效率。

## 3. 深度学习模型基本架构与模型定位

### 3.1 卷积神经网络的层次化特征

卷积神经网络通过局部连接和参数共享提取图像特征。浅层特征通常保留边缘、纹理与局部灰度变化；随着下采样加深，感受野扩大，特征逐渐表达更抽象的目标语义。分割任务既需要高层语义判断目标类别，也需要浅层空间细节确定边界，因此适合使用编码器—解码器结构。

### 3.2 U-Net 编码器—解码器结构

U-Net 包含收缩路径和扩张路径：编码器通过逐级下采样提取语义特征，解码器逐级上采样恢复分辨率。对应层之间使用跳跃连接，将编码器的高分辨率特征与解码器特征拼接，从而降低仅依赖深层特征造成的边界信息损失（见参考文献 [1]）。

### 3.3 EfficientNet-B0 编码器

EfficientNet 的核心思想是通过复合缩放同时协调网络深度、宽度和输入分辨率，而不是只扩大单一维度。其基础模块主要由 MBConv、深度可分离卷积、Squeeze-and-Excitation 通道注意力及残差连接组成。EfficientNet-B0 是该系列的基础规模模型，参数量较小，适合作为本项目的轻量级编码器（见参考文献 [2]）。

本实验不使用 EfficientNet-B0 原有的分类头，而是提取多个尺度的特征图并送入 U-Net 解码器。ImageNet 权重仅用于训练初始化；加载完整分割 checkpoint 进行评估或推理时，不重复加载预训练分类权重。

### 3.4 本模型在小组三模型中的定位

| 模型 | 输入方式 | 主要定位 | 本部分状态 |
|---|---|---|---|
| 2D U-Net + EfficientNet-B0 Encoder | 三维体数据按 z 轴切成二维切片 | 轻量、训练和推理速度较快、便于批量部署 | 本人负责，本文详细介绍 |
| nnU-Net v2 | 【由对应负责人补充】 | 强基线与自配置医学分割 | 【小组汇总时补充】 |
| MONAI SegResNet | 【由对应负责人补充】 | 三维卷积与体上下文建模 | 【小组汇总时补充】 |

该设计的主要优势是参数规模较小，可以直接复用成熟的二维预训练编码器；主要限制是每张切片独立建模，缺少显式的三维上下文，可能影响跨切片连续性和复杂边界的稳定性。

## 4. EfficientNet-B0 分割网络的层结构与参数设置

### 4.1 总体数据流

```mermaid
flowchart LR
    A["3D NIfTI 病例"] --> B["按 z 轴提取 2D 切片"]
    B --> C["百分位归一化并缩放至 512×512"]
    C --> D["EfficientNet-B0 多尺度编码"]
    D --> E["U-Net 解码与跳跃连接"]
    E --> F["1 通道 logits + Sigmoid"]
    F --> G["阈值 0.5 得到二值切片"]
    G --> H["恢复原尺寸并重建 3D mask"]
    H --> I["保留 affine/header 输出 NIfTI"]
```

### 4.2 编码器与解码器层级

以下结构由当前项目配置实际实例化后核对。输入尺寸按默认的 \(512\times512\) 计算：

| 层级 | 空间尺寸 | 输出通道 | 主要作用 |
|---|---:|---:|---|
| 输入 | 512×512 | 1 | 单通道灰度切片 |
| Encoder level 1 | 256×256 | 32 | 初始卷积与浅层纹理特征 |
| Encoder level 2 | 128×128 | 24 | MBConv 特征提取，作为跳跃连接 |
| Encoder level 3 | 64×64 | 40 | 扩大感受野，提取中层特征 |
| Encoder level 4 | 32×32 | 112 | 高层语义特征，作为跳跃连接 |
| Encoder bottleneck | 16×16 | 320 | 最深层语义表示 |
| Decoder block 1 | 32×32 | 256 | 上采样并融合 112 通道 skip feature |
| Decoder block 2 | 64×64 | 128 | 上采样并融合 40 通道 skip feature |
| Decoder block 3 | 128×128 | 64 | 上采样并融合 24 通道 skip feature |
| Decoder block 4 | 256×256 | 32 | 上采样并融合 32 通道 skip feature |
| Decoder block 5 | 512×512 | 16 | 恢复到输入分辨率 |
| Segmentation head | 512×512 | 1 | 3×3 卷积输出二分类 logits |

解码器默认通道序列为 `(256, 128, 64, 32, 16)`。跳跃连接使用通道拼接，将编码器的定位信息与解码器的高层语义特征融合。该通道序列来自当前 `segmentation-models-pytorch` U-Net 默认实现，而不是本项目自行设计的自定义 decoder（见参考文献 [3]）。

### 4.3 参数量

| 模块 | 可训练参数量 |
|---|---:|
| EfficientNet-B0 encoder | 4,006,972 |
| U-Net decoder | 2,243,776 |
| Segmentation head | 145 |
| **总计** | **6,250,893** |

当前模型全部参数均参与训练，总可训练参数量约为 6.25 M。该数值来自当前依赖版本和项目配置的实际模型实例，不使用分类头。

### 4.4 主要超参数

| 类别 | 参数 | 当前配置 | 说明 |
|---|---|---:|---|
| 数据 | 输入尺寸 | 512×512 | 图像双线性缩放，mask 最近邻缩放 |
| 数据 | 切片轴 | z | 将三维体数据转换为二维样本 |
| 数据 | 前景切片保留比例 | 1.0 | 保留全部含前景切片 |
| 数据 | 空切片保留比例 | 0.25 | 仅作用于训练集，降低背景主导程度 |
| 模型 | encoder | efficientnet-b0 | EfficientNet-B0 多尺度特征编码 |
| 模型 | encoder 初始化 | ImageNet | 下载失败时会警告并回退随机初始化 |
| 模型 | 输入/输出通道 | 1 / 1 | 灰度输入、二值输出 |
| 训练 | 随机种子 | 42 | 控制数据划分与训练随机性 |
| 训练 | epoch | 100 | 当前训练是否使用该值以最终日志为准 |
| 训练 | batch size | 8 | 需根据显存调整 |
| 训练 | 优化器 | AdamW | 当前仅支持 AdamW |
| 训练 | 学习率 | 0.0003 | 初始学习率 |
| 训练 | weight decay | 0.01 | 权重衰减 |
| 训练 | 损失函数 | Dice + BCE | 区域重叠与逐像素监督结合 |
| 训练 | AMP | true | 仅在 CUDA 设备上实际启用 |
| 推理 | 阈值 | 0.5 | Sigmoid 概率二值化阈值 |
| 推理 | slice batch size | 8 | 一个 forward 处理的切片数 |

最终稿应以本轮 `train.log` 中 `run_started.hyperparameters` 的实际值核对本表，特别是 epoch、batch size、设备、AMP 和预训练权重初始化状态。

## 5. 数据处理流程

### 5.1 病例级划分

项目默认使用随机种子 42，按病例进行 70%/15%/15% 的训练、验证和测试划分。同一病例的全部切片始终位于同一集合，从源头避免相邻切片泄漏导致的虚高指标。

当前病例集合尚未最终冻结。现有划分算法会先对全部病例重新洗牌，再按比例生成三个集合；如果后续新增或删除病例，即使随机种子不变，具体病例名单也可能变化。因此必须在数据冻结后重新记录最终划分和全部指标，不能直接复用旧数据版本的测试结果。

最终数据规模记录如下：

| 集合 | 病例数 | 二维切片数 | 含前景切片数 | 空切片数 |
|---|---:|---:|---:|---:|
| 训练集 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 |
| 验证集 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 |
| 测试集 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 |

### 5.2 掩膜二值化与切片采样

原始 mask 统一执行 `mask > 0` 转为 0/1。训练集保留全部含前景切片，并按 0.25 的比例确定性保留空切片；验证集和测试集保留全部切片，以保证评价覆盖完整病例。采样随机数由种子和病例编号共同确定，因此重复运行可复现。

### 5.3 强度归一化与尺寸变换

每张灰度切片使用第 1 和第 99 百分位进行强度截断与归一化：

\[
x'=\operatorname{clip}\left(\frac{x-P_1(x)}{P_{99}(x)-P_1(x)},0,1\right).
\]

当上下百分位相等时返回全零切片，避免除零。图像使用双线性插值缩放到 512×512，mask 使用最近邻插值，以避免插值产生非整数类别值；缩放后再次执行二值化。

## 6. 网络训练、验证与推理过程

### 6.1 训练过程

训练流程如下：

1. 加载 YAML 配置并校验模型 key、优化器、损失函数、epoch、batch size 和 worker 数。
2. 固定 Python、NumPy 和 PyTorch 随机种子；CUDA 可用时同步固定 CUDA 随机种子。
3. 按病例生成训练集与验证集，并建立 DataLoader。训练集开启 shuffle，验证集关闭 shuffle。
4. 使用 ImageNet 权重初始化 EfficientNet-B0 encoder；若预训练权重不可用，记录警告并回退到随机初始化。
5. 每个 batch 执行前向传播、Dice+BCE 损失计算、反向传播和 AdamW 参数更新。CUDA 环境可使用自动混合精度和 GradScaler（见参考文献 [4]、[5]）。
6. 每个 epoch 结束后，在完整验证病例上计算 Dice、IoU、Precision 和 Recall。
7. 若验证 Dice 高于历史最佳值，则原子保存新的 `best.pth`。

### 6.2 日志与 checkpoint

每次训练使用微秒精度 UTC 时间戳作为 `run_name`，例如 `20260710T083015123456Z`：

- 日志：`outputs/efficientnet_b0/<run_name>/train.log`
- 权重：`checkpoints/efficientnet_b0/<run_name>/best.pth`

日志为 JSONL 格式，主要事件包括 `run_started`、`epoch_completed`、`checkpoint_saved` 和 `run_completed`。首条日志保存模型、数据、优化器、学习率、batch size、epoch、AMP、设备、worker、验证阈值等主要超参数。checkpoint 保存模型状态、配置、最佳 epoch、最佳验证 Dice、初始化方式和 `run_name`，便于追溯实验。

### 6.3 验证方法

验证时先对同一病例的全部切片累计 TP、FP、FN、TN，再计算该病例的 Dice、IoU、Precision 和 Recall，最后对病例做宏平均。采用这种方式可以避免将每张切片错误地当作独立病例，也避免病例切片数不同对总体结果产生不合理的权重偏置。

### 6.4 推理过程

整卷 NIfTI 推理步骤如下：

1. 读取三维体数据，并检查输入是否为 3D。
2. 沿配置的 z 轴提取切片，按 `inference.batch_size` 分批前向传播。
3. 对每批 logits 使用 Sigmoid 和 0.5 阈值得到二值 mask。
4. 将每张 mask 用最近邻插值恢复到原始切片尺寸。
5. 按原顺序重建三维二值 mask。
6. 使用参考 NIfTI 保存结果，保留原影像的 affine 和 header。

统一批量推理接口同时支持 `.nii`、`.nii.gz` 和常见二维图片。NIfTI 输出命名为 `<case>_Seg.nii.gz`；二维图片输出同名二值 PNG。推理结果还记录病例编号、输出路径、体数据形状和端到端耗时。

## 7. 实验结果与对比分析

### 7.1 当前实验状态

本轮训练仍在进行，因此本节暂不填写最终指标，不根据中间 epoch 推断最终结论。仓库中存在旧流程的历史基线记录，但旧记录的测试病例与当前数据审计并不完全一致；加之本轮日志、checkpoint 命名及可能的训练配置已经更新，旧数字只能作为历史参考，不能直接写入本轮正式结果表。最终报告应使用数据冻结后的本轮完整训练与测试结果。

### 7.2 训练曲线

| 图表 | 文件/数值 | 当前状态 |
|---|---|---|
| Train loss–epoch 曲线 | 【待补充】 | 等待训练完成 |
| Validation Dice–epoch 曲线 | 【待补充】 | 等待训练完成 |
| Validation IoU–epoch 曲线 | 【待补充】 | 等待训练完成 |
| 最佳 epoch | 【待补充】 | 从 `checkpoint_saved` 日志提取 |
| 最佳验证 Dice | 【待补充】 | 从 `best.pth` 或日志提取 |

建议在最终稿插入两张图：一张同时展示 train loss 与 validation Dice 的趋势，另一张展示验证集 Dice/IoU/Precision/Recall。若训练损失继续下降而验证 Dice 持续下降，应在分析中说明可能存在过拟合。

### 7.3 EfficientNet-B0 最终结果

| 指标 | 验证集 | 测试集 |
|---|---:|---:|
| Dice | 【待补充】 | 【待补充】 |
| IoU | 【待补充】 | 【待补充】 |
| Precision | 【待补充】 | 【待补充】 |
| Recall | 【待补充】 | 【待补充】 |
| 平均推理时间/病例 | 【待补充】 | 【待补充】 |
| 峰值显存 | 【待补充】 | 【待补充】 |

逐病例结果：

| 病例 | 体数据尺寸 | Dice | IoU | Precision | Recall | 推理时间/s |
|---|---:|---:|---:|---:|---:|---:|
| 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 |
| 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 |

### 7.4 三模型横向对比

| 模型 | 参数量 | Dice | IoU | Precision | Recall | 推理时间/病例 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2D U-Net + EfficientNet-B0 | 6.25 M | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 |
| nnU-Net v2 | 【其他负责人补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 |
| MONAI SegResNet | 【其他负责人补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 | 【待补充】 |

最终分析应从以下方面展开：

1. **分割精度**：比较 Dice 和 IoU，判断不同模型对目标区域整体重叠的建模能力。
2. **误分与漏分**：结合 Precision 和 Recall 判断模型偏向过分割还是欠分割。
3. **二维与三维上下文**：分析 2D EfficientNet-B0 在跨切片连续性方面是否弱于三维模型。
4. **效率与资源占用**：结合参数量、推理耗时和峰值显存评价轻量化是否带来实际收益。
5. **数据规模影响**：说明测试病例数量及分布，避免将小样本上的高指标直接解释为稳定泛化能力。

### 7.5 可视化结果

最终稿建议至少放入：

- 预测较好病例的原图、真实 mask、预测 mask 和叠加图；
- 预测较差病例的同类可视化；
- 一处典型误分或漏分区域的局部放大图；
- 如条件允许，展示相邻切片以分析 2D 模型的跨切片连续性。

图片及病例说明：`【待补充】`

## 8. 个人完成的工作与可运行代码

### 8.1 个人完成的工作

本人在本项目中主要完成了以下工作：

1. 配置并实现 2D U-Net + EfficientNet-B0 encoder 分割模型。
2. 实现三维 NIfTI 按 z 轴生成二维训练样本的流程，包括 mask 二值化、空切片采样、百分位归一化和尺寸变换。
3. 实现 Dice+BCE 训练循环、病例级验证指标、AMP、最佳 checkpoint 保存及训练日志。
4. 将每次训练按 UTC 时间戳隔离，使用 `.pth` 保存权重，并在 JSONL 日志中记录主要超参数和逐轮指标。
5. 实现整卷切片批推理、三维 mask 重建、affine/header 保留及二维图片推理。
6. 将 EfficientNet-B0 接入公共批量推理接口，并补充相关自动化测试和运行文档。

### 8.2 主要代码文件

| 文件 | 作用 |
|---|---|
| `configs/efficientnet_b0/base.yaml` | EfficientNet-B0 模型、训练、推理及产物路径配置 |
| `src/img_seg/models/efficientnet_b0.py` | 模型构建、checkpoint 加载、NIfTI/图片预测 |
| `src/img_seg/training/efficientnet_b0_cli.py` | 训练、验证、日志、checkpoint 与评估 CLI |
| `src/img_seg/data/slices.py` | 二维切片样本、归一化、缩放和 NIfTI 缓存 |
| `src/img_seg/inference/batch.py` | 公共批量推理接口与结果组织 |
| `src/img_seg/inference/cli.py` | 命令行批量推理入口 |
| `src/img_seg/evaluation/metrics.py` | Dice、IoU、Precision、Recall 计算 |
| `tests/test_efficientnet_b0.py` | 模型、数据、训练日志和 checkpoint 测试 |
| `tests/test_inference_batch.py` | 批量推理与 checkpoint 枚举测试 |

### 8.3 环境与运行命令

安装依赖：

```powershell
uv sync
```

运行相关测试：

```powershell
uv run pytest tests/test_efficientnet_b0.py tests/test_inference_batch.py -q
uv run ruff check .
```

开始训练：

```powershell
uv run imgseg-efficientnet train --config configs/efficientnet_b0/base.yaml
```

训练完成后，控制台会输出实际 `run_name`、checkpoint 和日志路径。评估命令：

```powershell
$RUN_NAME = "【训练完成后填写】"
uv run imgseg-efficientnet evaluate `
  --config configs/efficientnet_b0/base.yaml `
  --checkpoint "checkpoints/efficientnet_b0/$RUN_NAME/best.pth" `
  --split test `
  --output-dir outputs/efficientnet_b0/test `
  --output-json outputs/efficientnet_b0/test/metrics.json
```

批量推理：

```powershell
$RUN_NAME = "【训练完成后填写】"
uv run imgseg-predict `
  --model efficientnet_b0 `
  --config configs/efficientnet_b0/base.yaml `
  --checkpoint "checkpoints/efficientnet_b0/$RUN_NAME/best.pth" `
  --input-dir Test `
  --output-dir Test_Seg
```

## 9. 个人对深度学习的理解和练习

通过本次实验，我对深度学习的理解从“调用一个网络完成预测”进一步转向了“建立可验证、可复现的完整实验系统”。模型结构只是结果的一部分，数据划分、预处理、损失函数、评价口径、日志和推理流程同样会直接影响结论是否可信。

第一，医学图像分割与普通图像分类不同。分类只需要给出整张图像的类别，而分割必须对每个像素作出判断。目标区域通常远小于背景，因此简单追求像素准确率可能得到“全部预测为背景”的无效模型。Dice、IoU 及 Dice 类损失能够更直接地反映目标区域的重叠质量。

第二，数据泄漏会使模型指标失去意义。三维体数据的相邻切片非常相似，如果先切片再随机分配，来自同一病例的切片可能同时出现在训练集和测试集，造成过于乐观的结果。因此本实验坚持先按病例划分，再在集合内部生成切片。

第三，模型设计本质上是精度、速度和资源之间的权衡。EfficientNet-B0 + U-Net 利用轻量级编码器和二维切片降低了显存与计算开销，但代价是缺少直接的三维上下文。是否值得采用该模型，不能只看 Dice，还应结合推理时间、显存、参数量和跨切片连续性进行判断。

第四，训练过程必须可追溯。将每次运行用时间戳隔离，并记录超参数、逐轮指标和最佳 checkpoint，可以避免不同实验相互覆盖，也能保证最终报告中的指标能够对应到具体配置和权重。

本次完成的主要练习包括：读取和保存 NIfTI、保持 affine/header、构造二维切片数据集、实现二值化和百分位归一化、搭建预训练编码器 U-Net、实现 Dice+BCE 损失、使用 AMP、按病例汇总指标、批量重建三维 mask，以及为训练和推理逻辑编写自动化测试。这些练习使我认识到，深度学习实验不仅需要理解网络原理，还需要具备数据工程、软件工程和实验设计能力。

## 10. 实验总结

本人完成了基于 EfficientNet-B0 编码器的 2D U-Net 医学图像二值分割流程。当前模型输入单通道 512×512 切片，总参数量约 6.25 M，采用 Dice+BCE 损失、AdamW 优化器和病例级验证。工程实现已经覆盖数据切片、训练日志、最佳 `.pth` checkpoint、整卷批量推理、空间信息保留及自动化测试。

当前训练仍在进行，因此暂不对最终精度和三模型优劣作确定性结论。训练结束后应补充完整曲线、最佳 epoch、验证/测试指标、逐病例结果、推理时间、峰值显存和可视化案例，再与 nnU-Net v2、MONAI SegResNet 在相同病例划分和评价口径下进行比较。最终结论需要同时讨论精度、效率、小样本可靠性以及二维模型缺少三维上下文的限制。

## 11. 参考文献

[1] O. Ronneberger, P. Fischer, and T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI, 2015. <https://arxiv.org/abs/1505.04597>

[2] M. Tan and Q. V. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML, 2019. <https://arxiv.org/abs/1905.11946>

[3] Segmentation Models PyTorch. *U-Net model documentation*. <https://smp.readthedocs.io/en/stable/models.html>

[4] PyTorch Documentation. *Automatic Mixed Precision Examples*. <https://docs.pytorch.org/docs/stable/notes/amp_examples.html>

[5] PyTorch Documentation. *AdamW*. <https://docs.pytorch.org/docs/stable/generated/torch.optim.adamw.AdamW_class.html>

## 12. 待补充数据清单

训练结束后，按以下顺序补充即可完成个人部分：

- [ ] 封面中的课程、小组、姓名、学号、班级、教师和日期。
- [ ] 本轮 `run_name`、日志路径和 checkpoint 外部存储位置。
- [ ] 训练/验证/测试病例数及各类切片数。
- [ ] 冻结当前数据版本并保存最终 train/val/test 病例名单。
- [ ] 实际 epoch、batch size、设备、AMP 和 encoder 初始化状态。
- [ ] Train loss、Validation Dice/IoU 曲线和最佳 epoch。
- [ ] 验证集与测试集 Dice、IoU、Precision、Recall。
- [ ] 每个测试病例的指标、体数据尺寸和推理时间。
- [ ] 峰值显存或至少注明未测量原因。
- [ ] 代表性成功/失败可视化及误差分析。
- [ ] nnU-Net v2、MONAI SegResNet 的结构与最终指标。
- [ ] 三模型在相同划分、相同指标口径下的对比结论。
- [ ] 代码压缩包、仓库地址或课程要求的附件形式。
