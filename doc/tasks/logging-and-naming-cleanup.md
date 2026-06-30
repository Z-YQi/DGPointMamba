# Logging And Naming Cleanup

## 这个文件是什么意思

这个模块专门处理当前代码中 DAPointMamba/DAMamba/UDA 命名残留、日志口径不一致、训练测试输出不利于后续分析的问题。它不改变模型能力，但会直接影响后续实验记录和日志分析。

## 需要确认的清理清单

- [x] `models/point_mamba.py` 中 active 日志前缀从 `[DAPointMamba]` 改为 `[DGPointMamba]`。
- [x] active 训练日志删除 `loss_sp` 和 `loss_ch`。
- [x] active 训练日志增加 `loss_rec_clean`、`loss_rec_aug_src`、`loss_total`。
- [x] generator 开启时记录 generator 类型。
- [x] Fixed SinPoint 记录 `A`、`w`、`rand_center_num`、`sample`。
- [ ] Learnable SinPoint/MSF 记录 `mean_abs_A`、`mean_abs_w`、`mean_abs_Aw`。
- [ ] TPDR 开启时记录 TPDR 分项 loss。
- [ ] DGDeformableScan 开启时记录 offset 和 token interchange 统计。
- [x] validation/test 输出表头固定为 `F-Score | CDL1 | CDL2 | EMDistance | UCD | UHD`。
- [x] 统一 `validate()` 和 `test()` 中 CDL1/CDL2 loss 的显示缩放口径。
- [x] TensorBoard tag 使用 DGPointMamba 命名，不再使用 `Sparse_L1`、`Sparse_L2` 这类不清楚的旧标签。
- [x] checkpoint 名称从 `ckpt_source_best` 改成 DG 语义名称。
- [x] 可视化输出目录从通用 `point_virtualization` 改成可配置的实验输出子目录。
- [x] 测试脚本默认配置从 `DAPointMamba.yaml` 改到 DGPointMamba 配置。
- [x] `utils/model_analyze.py` 不再引用不存在的 `DAMamba`。

## 完成标准

- [x] active 日志中没有 DAPointMamba/DAMamba/source-target adaptation 命名。
- [x] active 日志字段可以直接用于后续训练曲线和测试表格分析。
- [x] 所有 metric 缩放口径与 `doc/detailed-design.md` 一致。
- [ ] 旧命名如需保留，只能出现在历史说明文档或兼容注释中。
