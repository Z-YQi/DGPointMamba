# TPDR

## 这个文件是什么意思

这个模块实现 Topology-Preserving Deformation Regularization，用来约束 Learnable MSF 生成的伪域形变不要破坏局部拓扑和点级对应关系。

## 最小任务

- [ ] 新增 TPDR loss 计算函数或模块。
- [ ] 实现 displacement term：`L_disp = mean ||P_aug - P_s||_2`。
- [ ] 基于 `P_s` 构建 kNN 图。
- [ ] 实现 graph preservation term。
- [ ] 实现 homeomorphic bound term：`relu(abs(A_i * w_i) - rho)`。
- [ ] 支持从 Learnable MSF raw params 中读取 `A_i` 和 `w_i`。
- [ ] 添加配置字段 `tpdr.enable`、`tpdr.knn_k`、`tpdr.rho_homeo`。
- [ ] 添加 loss 权重：`tpdr`、`tpdr_disp`、`tpdr_graph`、`tpdr_homeo`。
- [ ] 将 `L_TPDR` 加入 total loss。
- [ ] 日志记录 `loss_tpdr`、`loss_tpdr_disp`、`loss_tpdr_graph`、`loss_tpdr_homeo`、`mean_Aw_violation`。

## 完成标准

- [ ] TPDR 关闭时不改变 Learnable MSF 的训练路径。
- [ ] TPDR 开启时所有 TPDR loss 都是 scalar finite tensor。
- [ ] kNN 图只从 `P_s` 构建。
- [ ] 可以跑出 Learnable MSF with TPDR 的 ablation 日志。
