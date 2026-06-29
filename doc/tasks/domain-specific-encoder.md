# Domain Specific Encoder

## 这个文件是什么意思

这个模块为 DGDeformableScan 提供 patch-level 几何域代码。它不能依赖 generator hint，因为推理时目标域没有 generator；它必须只从 tokens 和 patch centers 中估计 domain-specific geometric code。

## 最小任务

- [ ] 新增 `DomainSpecificEncoder` 模块。
- [ ] 输入 `tokens: [B, G, D]` 和 `centers: [B, G, 3]`。
- [ ] 实现 `DomainPosEmbed(centers)`，输出 `[B, G, D_dom]`。
- [ ] 将 `tokens` 和 `center_embed` concat。
- [ ] 用 MLP 输出 `domain_code_patch: [B, G, D_dom]`。
- [ ] 添加配置字段 `domain_specific_encoder.enable` 和 `domain_specific_encoder.domain_dim`。
- [ ] 支持关闭时返回 zero domain code。
- [ ] 在 clean branch 和 augmented branch 共享同一个 encoder。
- [ ] 日志或 aux 中可选记录 domain code norm。

## 完成标准

- [ ] clean 和 augmented 分支都能得到 `[B, G, D_dom]`。
- [ ] 推理时 `model(partial)` 不需要 generator hint 也能得到 domain code。
- [ ] 关闭该模块时后续 shape 不变。
- [ ] 该模块不引入 source-target 对齐 loss。
