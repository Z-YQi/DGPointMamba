# DGPointMamba Task Progress

## 使用方式

每个模块完成后，先更新对应模块文件中的 checklist，再勾选本文件中的模块级进度。模块完成标准是：对应文件的最小任务和完成标准都已满足，并且至少完成语法检查或可运行环境中的 smoke test。

## 总体进度

- [ ] `dg-training-runner.md` - Fixed SinPoint 双分支 DG 训练入口。
- [x] `backbone-cleanup.md` - 清理 UDA 双输入对齐骨架，统一单输入补全 backbone。
- [ ] `fixed-sinpoint-baseline.md` - 第一条正式 Fixed SinPoint baseline。
- [ ] `learnable-global-sinpoint.md` - 可学习全局 SinPoint baseline。
- [ ] `learnable-msf-generator.md` - 可学习 multi-anchor sine field 伪域生成器。
- [ ] `tpdr.md` - 拓扑保持形变正则。
- [ ] `domain-specific-encoder.md` - patch-level 几何域代码编码器。
- [ ] `dg-deformable-scan.md` - Mamba 前的形变感知扫描模块。
- [ ] `configs-and-ablation.md` - DGPointMamba 配置和消融入口。
- [ ] `logging-and-naming-cleanup.md` - DG 命名、日志、metric 口径清理。
- [ ] `experiment-logging-and-analysis.md` - 实验结构化日志、汇总表和结果分析流程。
- [ ] `remote-git-workflow.md` - 本地 Git、远程训练和日志回传流程。

## 推荐执行顺序

1. `backbone-cleanup.md`
2. `fixed-sinpoint-baseline.md`
3. `dg-training-runner.md`
4. `logging-and-naming-cleanup.md`
5. `experiment-logging-and-analysis.md`
6. `configs-and-ablation.md`
7. `remote-git-workflow.md`
8. `learnable-global-sinpoint.md`
9. `learnable-msf-generator.md`
10. `tpdr.md`
11. `domain-specific-encoder.md`
12. `dg-deformable-scan.md`
