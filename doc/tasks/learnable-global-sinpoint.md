# Learnable Global SinPoint

## 这个文件是什么意思

这个模块把 Fixed SinPoint 的全局振幅和频率改成可学习参数，作为比 Fixed SinPoint 更强但仍然简单的第二个 baseline。

## 最小任务

- [ ] 新增 `LearnableSinPoint` 模块，继承 `torch.nn.Module`。
- [ ] 使用 `A_raw` 和 `w_raw` 作为可学习参数。
- [ ] 用 `A = A_max * tanh(A_raw)` 限制振幅范围。
- [ ] 用 `w = w_max * tanh(w_raw)` 限制频率范围。
- [ ] 支持参数形状 `[3]` 或 `[1, 1, 3]`，并与 `[B, N, 3]` 广播兼容。
- [ ] 复用 Fixed SinPoint 的归一化和反归一化思路。
- [ ] 输出 `P_aug` 和包含 `A`、`w`、`mean_abs_A`、`mean_abs_w`、`mean_abs_Aw` 的 `stats`。
- [ ] 添加配置字段 `domain_generator.type: learnable_global_sinpoint`。
- [ ] 将 generator 参数加入 optimizer。
- [ ] 在训练日志和 TensorBoard 中记录 learnable SinPoint 统计量。

## 完成标准

- [ ] Learnable global SinPoint baseline 与 Fixed SinPoint 使用同一套双分支损失。
- [ ] `A`、`w` 和 `A*w` 统计量为 finite。
- [ ] `P_aug` 保持 `[B, N, 3]`。
- [ ] 可以和 Fixed SinPoint baseline 直接比较 3D-FUTURE CDL2。
