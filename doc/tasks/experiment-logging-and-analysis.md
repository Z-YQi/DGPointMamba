# Experiment Logging And Analysis

## 这个文件是什么意思

这个任务负责把 DGPointMamba 的实验记录做成可分析、可比较、可复现的格式。它不负责改模型结构；它负责让每次训练和测试都能留下足够信息，后续可以判断指标变化来自哪个模块、哪个 loss、哪个超参数或哪个数据设置。

该任务适合交给专门的实验分析 agent。实验分析 agent 默认只读日志、配置和结果文件，不修改模型代码，除非主 agent 明确要求。

## 允许修改

- 训练和测试日志相关代码。
- TensorBoard tag 命名相关代码。
- 实验输出目录命名和 summary 写入逻辑。
- 新增日志解析、汇总、画图脚本。
- 新增 `metrics.jsonl`、`summary.csv` 写入工具。
- 新增实验分析说明文档或报告模板。

## 禁止事项

- 不得改变模型 forward 语义。
- 不得改变 loss 定义。
- 不得改变 validation/test metric 计算逻辑。
- 不得改变数据集划分。
- 不得把目标域训练数据引入训练。
- 不得为了日志方便重命名已有 checkpoint 导致旧实验不可追踪。

## 最小任务

- [ ] 为每个实验记录 `experiment_name`、`commit_hash`、`config_path`、`seed`、`source_dataset`、`target_dataset`、`category`、`output_dir`、`checkpoint_path`。
- [ ] 训练阶段写入 step 或 epoch 级 `metrics.jsonl`。
- [ ] 每次验证或测试后更新 one-row-per-experiment 的 `summary.csv`。
- [ ] 记录基础 loss：`loss_total`、`loss_rec_clean`、`loss_rec_aug_src`。
- [ ] 当模块开启时记录对应 loss：`loss_tpdr`、`loss_tpdr_disp`、`loss_tpdr_graph`、`loss_tpdr_homeo`、`loss_offset`。
- [ ] 当 generator 或 DGDeformableScan 开启时记录统计量：`mean_abs_A`、`mean_abs_w`、`mean_abs_Aw`、`mean_Aw_violation`、`mean_delta_p`、`mean_delta_t`、`token_interchange_ratio`、`domain_code_norm`。
- [ ] 记录 evaluation metrics，顺序固定为 `F-Score | CDL1 | CDL2 | EMDistance | UCD | UHD`。
- [ ] 保证 TensorBoard tag 与 `metrics.jsonl` 字段名尽量一致。
- [ ] 新增一个轻量日志分析脚本，能读取多个实验输出目录并生成对比表。
- [ ] 新增一个 loss 趋势分析入口，至少能比较 `loss_total`、`loss_rec_clean`、`loss_rec_aug_src` 和主 metric。
- [ ] 为失败实验记录 `status: failed`、失败阶段、错误摘要和最后可用 step。

## 推荐输出文件

每个实验输出目录建议包含：

```text
run_meta.json
metrics.jsonl
summary.csv
train.log
test.log
config.yaml
checkpoints/
tb/
analysis/
```

`run_meta.json` 推荐字段：

```json
{
  "experiment_name": "fixed_sinpoint_crn_to_3dfuture_cabinet_seed1",
  "commit_hash": "",
  "config_path": "",
  "seed": 1,
  "source_dataset": "CRN",
  "target_dataset": "3D-FUTURE",
  "category": "cabinet",
  "method": "fixed_sinpoint",
  "output_dir": "",
  "checkpoint_path": "",
  "status": "running"
}
```

`summary.csv` 推荐列：

```text
experiment_name,commit_hash,config_path,seed,source_dataset,target_dataset,category,method,status,best_epoch,best_metric_name,best_metric_value,F-Score,CDL1,CDL2,EMDistance,UCD,UHD,checkpoint_path,output_dir
```

## 分析规则

分析实验时先回答这些问题：

- 当前实验和对照实验是否使用同一个 commit、seed、数据集、category 和训练轮数。
- 主 metric 是否改善，早期开发优先看 3D-FUTURE CDL2。
- `loss_rec_clean` 和 `loss_rec_aug_src` 是否同步下降。
- 如果 `loss_rec_aug_src` 明显高于 `loss_rec_clean`，检查 generator 是否过强。
- 如果 TPDR 开启后性能下降，检查 `loss_tpdr_*` 是否过大，优先降低 `lambda_tpdr` 或生成器强度。
- 如果 DGDeformableScan 开启后不稳定，检查 `mean_delta_p`、`mean_delta_t` 是否长期贴近 clamp 边界。
- 如果训练 loss 正常但 test metric 异常，优先检查 metric 缩放、数据集 split、checkpoint 加载和 evaluation config。

## 完成标准

- [ ] 每个正式实验都能生成 `run_meta.json`、`metrics.jsonl` 和 `summary.csv`。
- [ ] Fixed SinPoint baseline 至少能记录 clean/aug reconstruction loss 和 3D-FUTURE CDL2。
- [ ] 多个实验可以按 `summary.csv` 横向比较。
- [ ] 实验分析 agent 能根据日志判断 loss 趋势和主要异常来源。
- [ ] 失败实验不会只留下散乱终端输出，而是能被后续复盘。

## 给实验分析 agent 的提示词模板

```text
请先阅读 AGENTS.md、doc/proposal.md、doc/detailed-design.md 和 doc/tasks/experiment-logging-and-analysis.md。
这次只分析实验日志，不修改模型代码。
请读取我提供的 run_meta.json、metrics.jsonl、summary.csv、train.log、test.log 和 config。
请回答：
1. 主指标相比对照是否改善
2. loss_total、loss_rec_clean、loss_rec_aug_src 的趋势是否正常
3. 是否有 NaN/Inf、震荡、过拟合或 generator 过强迹象
4. 最可能影响结果的模块或超参数
5. 下一步只改一个变量时应该改什么
```

