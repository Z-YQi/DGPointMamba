# Learnable MSF Generator

## 这个文件是什么意思

这个模块实现主要的可学习伪域生成器：multi-anchor sine fields。它把单一全局正弦扰动扩展成多个 anchor-conditioned 局部形变场，用于产生更有结构的源域伪域。

## 最小任务

- [ ] 新增 `LearnableMSFDomainGenerator` 模块。
- [ ] 输入 `P_s: [B, N, 3]`，输出 `P_aug: [B, N, 3]`。
- [ ] 第一版使用全局可学习参数 `A_i: [k, 3]`、`w_i: [k, 3]`、`phi_i: [k, 3]`。
- [ ] 使用 FPS 选择 `k` 个 anchors。
- [ ] 实现 anchor-conditioned sine field。
- [ ] 在 normalized point cloud 上应用形变，再恢复原尺度。
- [ ] 不改变点数和点顺序，保留点级对应关系。
- [ ] 生成 `domain_hint_patch: [B, G, D_dom]` 作为日志或后续 ablation 输入。
- [ ] 生成 `domain_hint_global: [B, D_dom]` 作为日志或后续 ablation 输入。
- [ ] 输出 raw 参数和 deformation magnitude 统计。
- [ ] 添加配置字段 `domain_generator.type: learnable_msf`。
- [ ] 暂不实现 input-conditioned 参数，保留配置开关但默认 false。

## 完成标准

- [ ] Learnable MSF 可以替换 Learnable global SinPoint 进入同一训练 runner。
- [ ] `P_aug`、`domain_hint_patch`、`domain_hint_global` shape 与设计一致。
- [ ] 形变统计为 finite。
- [ ] 仍然只用原始 `Y_s` 做监督。
