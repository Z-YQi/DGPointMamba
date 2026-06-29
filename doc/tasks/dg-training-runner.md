# DG Training Runner

## 这个文件是什么意思

这个模块负责把当前 DAPointMamba 风格的训练入口改成 DGPointMamba 的第一条可运行训练路径。第一阶段不做无增强 source-only 对比实验，而是直接从 Fixed SinPoint baseline 开始：同一个源域 batch 生成 clean branch 和 augmented branch，训练损失为 `L_rec_clean + lambda_aug * L_rec_aug_src`。

## 最小任务

- [ ] 训练阶段只构造 CRN/ShapeNet 源域 train dataloader，不构造目标域 train dataloader。
- [ ] 保留目标域 validation/test dataloader，用于 CRN -> 3D-FUTURE、ModelNet、KITTI、ScanNet、MatterPort3D 的评估。
- [ ] 统一源域 batch 解析为 `P_s` 和 `Y_s`，避免使用会误删 batch 维度的裸 `squeeze()`。
- [ ] 通过配置选择 Fixed SinPoint generator，生成 `P_aug`。
- [ ] clean branch 调用 `model(P_s)` 得到 `pred_clean`。
- [ ] augmented branch 调用 `model(P_aug)` 得到 `pred_aug`。
- [ ] 使用原始 `Y_s` 同时监督 `pred_clean` 和 `pred_aug`，不生成 deformed complete target。
- [ ] 计算并记录 `loss_rec_clean`、`loss_rec_aug_src`、`loss_total`。
- [ ] 删除训练路径里的 `target_partial = aug_partial` 覆盖逻辑。
- [ ] 删除训练路径里的 `loss_sp`、`loss_ch` 参与总损失逻辑。
- [ ] 保持 `step_per_update`、optimizer、scheduler、checkpoint 保存逻辑可用。
- [ ] 为 CPU 不可用 CUDA 的本地环境保留至少 `py_compile` 级别的轻量检查。

## 完成标准

- [ ] `tools/runner.py` 中训练阶段没有 active target train dataloader。
- [ ] Fixed SinPoint baseline 的第一轮训练日志能看到 clean/aug 两个重建损失。
- [ ] 目标域 validation/test 仍能按原来的 metric 表输出。
- [ ] active training log 中不再出现 `loss_sp` 或 `loss_ch`。
