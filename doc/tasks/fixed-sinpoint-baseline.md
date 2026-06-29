# Fixed SinPoint Baseline

## 这个文件是什么意思

这个模块是第一条正式消融实验。它使用当前固定参数 SinPoint 生成源域伪域输入 `P_aug`，然后和 clean branch 一起训练共享补全 backbone。

## 最小任务

- [ ] 保留当前 `utils/sinpoint_augmentation.py` 作为固定增强实现。
- [ ] 封装一个统一 generator 调用接口，输入 `P_s`，输出 `P_aug` 和 `stats`。
- [ ] 添加配置字段 `domain_generator.type: fixed_sinpoint`。
- [ ] 添加 Fixed SinPoint 参数配置：`A`、`w`、`rand_center_num`、`sample`、`isCat`、`shuffle`。
- [ ] 默认 `isCat: false`，确保 `P_aug` 点数和 batch size 与 `P_s` 一致。
- [ ] 确保 `P_aug.device == P_s.device`。
- [ ] 确保 `P_aug.shape == P_s.shape`。
- [ ] 将 Fixed SinPoint 接入 `DG Training Runner` 的 augmented branch。
- [ ] 日志记录 Fixed SinPoint 参数和增强幅度基础统计。
- [ ] 不对 `Y_s` 做同步形变。

## 完成标准

- [ ] Fixed SinPoint baseline 可以跑到第一次 validation。
- [ ] 训练总损失为 `loss_rec_clean + lambda_aug * loss_rec_aug_src`。
- [ ] 该 baseline 不使用目标域训练 batch。
- [ ] 该 baseline 不使用 source-target alignment loss。
