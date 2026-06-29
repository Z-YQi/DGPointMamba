# Configs And Ablation

## 这个文件是什么意思

这个模块负责把实验配置从 DAPointMamba 风格整理成 DGPointMamba 风格，并为后续逐步加入模块的消融实验提供稳定入口。

## 最小任务

- [ ] 新增或更新 DGPointMamba 主配置，模型名统一为 `DGPointMamba`。
- [ ] 所有 active 配置使用 `mamba_depth: 12`。
- [ ] 删除 active 配置中的 `mamba_config.depth` 和 `decoder_depth` 双深度表达。
- [ ] 第一版所有 active 配置使用 `order_mode: z`。
- [ ] 删除 active 配置中的 `lambda_spatial` 和 `lambda_channel`。
- [ ] 添加 `domain_generator` 配置块。
- [ ] 添加 `domain_specific_encoder` 配置块。
- [ ] 添加 `dg_deformable_scan` 配置块。
- [ ] 添加 `tpdr` 配置块。
- [ ] 添加 `loss_weights` 配置块。
- [ ] 准备 Fixed SinPoint baseline 配置。
- [ ] 准备 Learnable global SinPoint 配置。
- [ ] 准备 Learnable MSF without TPDR 配置。
- [ ] 准备 Learnable MSF with TPDR 配置。
- [ ] 准备 Full DGPointMamba 配置。
- [ ] 明确 early ablation 优先使用 CRN -> 3D-FUTURE cabinet。

## 完成标准

- [ ] active 配置不会引用未注册的 `DAPointMamba` 或 `DAMamba`。
- [ ] 每个 ablation 配置只打开它对应的模块。
- [ ] 配置文件名和实验名能看出模块开关。
- [ ] validation metric 选择与目标域一致。
