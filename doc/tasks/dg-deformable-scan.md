# DG Deformable Scan

## 这个文件是什么意思

这个模块把 domain-specific geometry 注入 Mamba 前的 patch token 序列。它包含 scan-aware offset prediction、position embedding 更新、可选 token interchange，是完整 DGPointMamba 的核心结构改造。

## 最小任务

- [ ] 新增 `DGDeformableScan` 模块。
- [ ] 输入 `tokens: [B, G, D]`、`centers: [B, G, 3]`、`domain_code_patch: [B, G, D_dom]`。
- [ ] 实现 base position embedding。
- [ ] 拼接 `tokens`、`pos_base`、`domain_code_patch` 作为 offset 输入。
- [ ] 用 OffsetNet 输出 `offsets: [B, G, 4]`。
- [ ] 将 `offsets[..., 0:3]` 作为 `delta_p` 并按 `max_delta_p` clamp。
- [ ] 将 `offsets[..., 3:4]` 作为 `delta_t` 并按 `max_delta_t` clamp。
- [ ] 计算 `centers_def = centers + delta_p`，不覆盖原始 centers。
- [ ] 用 `centers_def` 生成 `pos_scan: [B, G, D]`。
- [ ] 用 `delta_t` 生成 order bias 并加到 tokens。
- [ ] 第一版 token interchange 默认关闭。
- [ ] 实现 token interchange 开关，开启时使用 stop-gradient donor。
- [ ] 添加 offset regularization：`L_offset`。
- [ ] 日志记录 `mean_delta_p`、`mean_delta_t`、`loss_offset`、`token_interchange_ratio`。

## 完成标准

- [ ] 模块输出 `tokens_scan: [B, G, D]`、`pos_scan: [B, G, D]`、`offsets: [B, G, 4]`。
- [ ] 禁用 offsets 时输出 shape 不变。
- [ ] 禁用 token interchange 时输出 shape 不变。
- [ ] DGDeformableScan 关闭时 plain Mamba stack 路径仍可运行。
