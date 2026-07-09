## 改动概述

-

## 运行与验证

- [ ] `uv run pytest`
- [ ] `uv run ruff check .`
- [ ] 相关 smoke test：

如未运行，请说明原因：

## 数据、模型与实验影响

- [ ] 不涉及数据、训练或评估结果
- [ ] 已更新 `docs/EXPERIMENTS.md`
- [ ] 已更新相关配置或文档

说明：

## 检查清单

- [ ] 已阅读并遵守 `AGENTS.md`
- [ ] 没有提交 `dataset/**/*.nii*`、`checkpoints/`、`outputs/`、`Test_Seg/` 或 `.venv/`
- [ ] 训练/验证/测试划分按 case 处理，没有 slice 级混入
- [ ] WebUI 改动仍通过统一推理接口调用模型
- [ ] 已说明已知风险、限制或后续工作
