# Remote Git Workflow

## 这个文件是什么意思

这个任务负责固化本地 Codex、Git 仓库和远程训练服务器之间的工作流。它不直接提升模型指标，但能保证每次实验都有明确代码版本、配置、日志和 checkpoint，避免本地和远程代码不一致导致结果不可复现。

该任务适合由主 agent 维护。暂时不建议交给独立 agent 自动执行破坏性 Git 或远程命令；远程/Git agent 可以负责检查和生成命令，但涉及覆盖、删除、reset、force push、覆盖 checkpoint 的操作必须人工确认。

## 允许修改

- 新增或更新远程运行说明文档。
- 新增安全的同步脚本或命令模板。
- 新增实验目录命名规范。
- 新增远程训练命令模板。
- 新增 Git 提交和 tag 规范。
- 新增日志回传和 summary 汇总流程。

## 禁止事项

- 不得执行 `git reset --hard`、强制 checkout、强制 push，除非用户明确要求。
- 不得删除远程 checkpoint 或实验输出目录，除非用户明确确认。
- 不得在远程服务器直接长期改代码而不同步回 Git。
- 不得让远程代码处于未记录 diff 的状态就开始正式实验。
- 不得覆盖已有实验目录。
- 不得把大型数据集、checkpoint、TensorBoard event 文件提交到 Git。

## 最小任务

- [ ] 确认本地目录是有效 Git 仓库，能执行 `git status`。
- [ ] 明确本地 repo 和远程 repo 的对应关系。
- [ ] 明确远程服务器上的项目路径、conda 环境、数据集路径和输出路径。
- [ ] 写清楚本地改代码后的提交流程。
- [ ] 写清楚远程拉取代码后的训练流程。
- [ ] 写清楚远程训练完成后日志和 summary 如何同步回本地。
- [ ] 写清楚 checkpoint 的保留策略和命名规则。
- [ ] 写清楚失败实验如何记录。
- [ ] 写清楚哪些目录必须加入 `.gitignore`。
- [ ] 为 Fixed SinPoint baseline 写出第一条远程训练命令模板。

## 推荐本地流程

每个模块开发完成后：

```text
1. 检查当前 diff。
2. 运行可用的本地轻量检查。
3. 更新对应 doc/tasks/*.md checklist。
4. 请主 agent review。
5. 修正 review 问题。
6. commit。
7. push 到远程 Git 仓库。
```

推荐 commit 粒度：

```text
一个任务文件对应一个或少数几个 commit。
不要把多个研究模块混在一个 commit。
不要把纯文档、模型结构、训练 runner、日志系统的大改混成一个 commit。
```

## 推荐远程流程

远程服务器只负责训练、测试和 CUDA 编译。

```text
1. 登录远程服务器。
2. 进入项目目录。
3. 确认当前没有未提交的远程修改。
4. 拉取指定 commit。
5. 激活训练环境。
6. 确认数据集路径。
7. 启动训练，并把 stdout/stderr 保存到 train.log。
8. 测试 best checkpoint，并保存 test.log。
9. 同步 run_meta.json、metrics.jsonl、summary.csv、train.log、test.log 和必要的 config 回本地。
10. 不同步大型 checkpoint，除非当前分析需要。
```

每次正式实验都必须记录：

```text
commit_hash
config_path
experiment_name
seed
source_dataset
target_dataset
category
train_command
test_command
output_dir
checkpoint_path
best_metric
```

## 实验目录命名

推荐格式：

```text
outputs/dgpointmamba/<method>/<source>_to_<target>/<category>/seed<seed>/<run_id>/
```

例子：

```text
outputs/dgpointmamba/fixed_sinpoint/crn_to_3dfuture/cabinet/seed1/2026-06-29_001/
```

## Fixed SinPoint baseline 命令模板

实际命令需要根据当前项目训练脚本参数调整。模板如下：

```bash
python main.py \
  --config cfgs/DGPointMamba_fixed_sinpoint.yaml \
  --exp_name fixed_sinpoint_crn_to_3dfuture_cabinet_seed1 \
  --seed 1
```

如果项目使用 launcher 或分布式脚本，应保持同样的 config、exp name 和 seed 规则。

## 日志回传规则

优先回传小文件：

```text
run_meta.json
metrics.jsonl
summary.csv
train.log
test.log
config.yaml
analysis/
```

默认不回传：

```text
完整数据集
大型 checkpoint
TensorBoard event 大文件
临时缓存
编译产物
```

只有需要复现、可视化或继续训练时，才同步指定 checkpoint。

## Git 和远程异常处理

如果本地 `git status` 不可用：

- 先不要开始正式代码修改。
- 检查 `.git` 是否完整。
- 如果本目录应该来自远程仓库，优先重新 clone。
- 如果本目录是普通源码目录，先初始化 Git 并设置远程仓库。

如果远程代码和本地 commit 不一致：

- 不要直接开始训练。
- 先确认远程 diff。
- 将远程必要修改整理成 patch 或 commit。
- 由主 agent 判断是否合并。

如果训练中断：

- 保留已有 `train.log`。
- 在 `run_meta.json` 或失败记录中标记 `status: failed`。
- 记录中断 epoch、最后 step、错误摘要。
- 不覆盖该输出目录，重新运行时使用新的 run id。

## 完成标准

- [ ] 本地和远程代码版本能通过 commit hash 对齐。
- [ ] Fixed SinPoint baseline 有明确远程运行命令模板。
- [ ] 每个实验输出目录不会互相覆盖。
- [ ] 日志和 summary 能从远程同步回本地用于分析。
- [ ] Git 不跟踪大型数据、checkpoint、缓存和临时输出。
- [ ] 失败实验有记录，不会污染成功实验结果。

## 给远程/Git 检查 agent 的提示词模板

```text
请先阅读 AGENTS.md 和 doc/tasks/remote-git-workflow.md。
这次只检查本地/远程/Git 流程，不修改模型代码。
请检查当前 Git 状态、需要忽略的目录、实验输出命名、远程训练命令模板和日志回传流程。
涉及删除、reset、force push、覆盖远程文件的操作不要执行，只给出风险说明和建议命令。
```

