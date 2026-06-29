# Backbone Cleanup

## 这个文件是什么意思

这个模块负责把模型主体从 DAPointMamba 的 UDA 双输入对齐骨架清理成 DGPointMamba 的单输入补全骨架。它是后续所有 generator、TPDR、DGDeformableScan 的共同基础。

## 最小任务

- [ ] 将 active 模型日志前缀从 `DAPointMamba` 改为 `DGPointMamba`。
- [ ] 将模型 forward 主路径统一为 `model(partial)`。
- [ ] 让 forward 返回预测结果和可选 `aux` 信息，训练/验证/测试调用方同步适配。
- [ ] 移除或停用 `SpatialSSM` 和 `ChannelSSM` 在 active forward 中的 source-target alignment 逻辑。
- [ ] 删除 active forward 输出中的 `loss_sp`、`loss_ch`。
- [ ] 删除未使用的 `self.blocks`，避免 `mamba_config.depth` 误导。
- [ ] 将 active Mamba 层数统一由 `mamba_depth` 控制。
- [ ] 将 `MambaDecoder` 或等价 Mamba stack 的层数改读 `config.mamba_depth`。
- [ ] 将 `order_mode` 实际传入 `PatchGroup`。
- [ ] 第一版 `order_mode` 只支持 `z`，配置和实现保持一致。
- [ ] 保证 decoder 输入输出形状保持 `[B, M, 3]`。

## 完成标准

- [ ] 模型中只有一个 active Mamba depth 字段：`mamba_depth`。
- [ ] `model(partial)` 可用于训练、验证和测试。
- [ ] active 模型路径不需要 `pts_t`。
- [ ] `order_mode: z` 不是被忽略的配置项。
