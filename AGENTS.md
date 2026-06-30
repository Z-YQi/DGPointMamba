# DGPointMamba Agent Rules

This repository is being refactored and extended for DGPointMamba experiments.
The goal is to improve domain-generalized point cloud completion metrics while
keeping every experiment reproducible, debuggable, and easy to roll back.

本文件是所有 Codex 会话的项目规则。新的开发会话、审查会话、实验分析会话都必须先阅读本文件，再阅读对应任务文档。

## Required Reading

Every agent must read these files before making code changes:

1. `AGENTS.md`
2. `doc/proposal.md`
3. `doc/detailed-design.md`
4. The relevant task file under `doc/tasks/`

For progress tracking, also read:

- `doc/tasks/progress.md`

Do not rely on chat history as the source of truth. Treat the repository documents as the long-term memory.

## Agent Roles

### Main Agent

The current main conversation acts as the main agent.

Responsibilities:

- Maintain project rules and task documents.
- Decide the next module to implement.
- Review diffs from module-development sessions.
- Check whether a change matches the task file and detailed design.
- Before approving push, remote smoke test, or formal experiment, verify that all prerequisite tasks in `doc/tasks/progress.md` are complete or explicitly waived.
- Analyze experiment logs and decide follow-up experiments.
- Keep the implementation path conservative and staged.

The main agent should not approve a new module until the previous module has a usable checkpoint or a clearly documented failure.
The main agent must not approve an experiment run when an earlier prerequisite task in the planned order is still incomplete, unless the waiver is written down with a concrete reason and risk.

### Module Development Agent

A module-development agent should work on exactly one task file.

Responsibilities:

- Read the required documents.
- Inspect the current code before editing.
- Before implementation, perform a call-path audit covering training data flow, loss data flow, and model forward return values.
- Implement only the requested task.
- Avoid unrelated refactors.
- Run lightweight local checks when full training is unavailable.
- Summarize changed files, validation results, and remaining risks.

Rules:

- Do not implement future-stage modules early.
- Do not change experiment semantics outside the current task.
- Do not silently change metric order, loss definitions, or dataset splits.
- Do not introduce target-domain training loss or source-target alignment loss.
- If the design document and existing helper function semantics disagree, stop and report the mismatch instead of guessing.
- Every completed checklist item must be supported by file and line-number evidence in the final summary.

### Experiment Analysis Agent

An experiment-analysis agent should read logs and configs, not modify model code unless explicitly asked.

Responsibilities:

- Parse training logs, `metrics.jsonl`, TensorBoard exports, and `summary.csv`.
- Compare experiments by commit hash, config, seed, dataset, and checkpoint.
- Analyze loss trends and metric changes.
- Identify whether a regression is likely caused by reconstruction loss, generator strength, TPDR, offsets, data issues, or evaluation mismatch.
- Recommend the next small hyperparameter change.
- Follow `doc/tasks/experiment-logging-and-analysis.md`.

### Remote/Git Workflow

Do not let a separate agent freely perform destructive Git or remote operations.
Remote training and Git synchronization should follow documented commands and require human confirmation when they can overwrite work.

The preferred workflow is:

- Local machine: code editing, documentation, config preparation, log analysis.
- Remote server: training, testing, CUDA extension compilation.
- Git: synchronize local and remote repositories with explicit commits.

Follow `doc/tasks/remote-git-workflow.md` for concrete command templates and safety rules.

## Development Order

Follow `doc/tasks/progress.md` unless the main agent decides otherwise.

Current recommended order:

1. `doc/tasks/backbone-cleanup.md`
2. `doc/tasks/fixed-sinpoint-baseline.md`
3. `doc/tasks/dg-training-runner.md`
4. `doc/tasks/logging-and-naming-cleanup.md`
5. `doc/tasks/experiment-logging-and-analysis.md`
6. `doc/tasks/configs-and-ablation.md`
7. `doc/tasks/remote-git-workflow.md`
8. `doc/tasks/learnable-global-sinpoint.md`
9. `doc/tasks/learnable-msf-generator.md`
10. `doc/tasks/tpdr.md`
11. `doc/tasks/domain-specific-encoder.md`
12. `doc/tasks/dg-deformable-scan.md`

The first formal experiment starts from Fixed SinPoint baseline. A no-generator source-only setting may still be kept as a debugging sanity configuration, but it is not the first formal ablation target.

## Core Research Rules

The active DGPointMamba path must follow these rules:

- Training uses source-domain training data only.
- Target-domain data is used only for validation and test.
- `Y_s` is not deformed in the main method.
- Pseudo-domain augmentation produces `P_aug` from `P_s`.
- Clean branch predicts from `P_s`.
- Augmented branch predicts from `P_aug`.
- Both branches are supervised by the original `Y_s`.
- No active source-target alignment loss is allowed.
- Old `loss_sp` and `loss_ch` must not contribute to the active total loss.
- Active logs should use DGPointMamba naming, not DAPointMamba, DAMamba, or UDA naming.

Main loss:

```text
L_total =
  L_rec_clean
  + lambda_aug * L_rec_aug_src
  + lambda_tpdr * L_TPDR
  + lambda_offset * L_offset
```

Only include `L_TPDR` and `L_offset` when their modules are enabled.

## Shape Contracts

Preserve the shape contracts in `doc/detailed-design.md`.

Important defaults:

- Input partial point cloud: `P_s: [B, N, 3]`
- Complete target: `Y_s: [B, M, 3]`
- Augmented partial point cloud: `P_aug: [B, N, 3]`
- Patch tokens: `tokens: [B, G, D]`
- Patch centers: `centers: [B, G, 3]`
- Domain code: `domain_code_patch: [B, G, D_dom]`
- DGDeformableScan offsets: `offsets: [B, G, 4]`

Avoid unsafe bare `squeeze()` calls on batch tensors because they can remove the batch dimension when `B == 1`.

## Configuration Rules

Active DGPointMamba configs should:

- Use `model.NAME: DGPointMamba`.
- Use one active Mamba depth field: `mamba_depth`.
- Avoid duplicate active depth fields such as `mamba_config.depth` and `decoder_depth`.
- Use `order_mode: z` in the first stable version.
- Use explicit module switches:
  - `domain_generator`
  - `domain_specific_encoder`
  - `dg_deformable_scan`
  - `tpdr`
  - `loss_weights`

Each ablation config should enable only the modules needed for that ablation.

Recommended formal ablations:

- Fixed SinPoint baseline
- Learnable global SinPoint
- Learnable MSF without TPDR
- Learnable MSF with TPDR
- Full DGPointMamba

Recommended debugging-only config:

- Generator disabled source-only sanity run

## Logging Requirements

All experiment logs should be useful for later analysis.

Record experiment metadata:

- experiment name
- commit hash
- config path
- seed
- source dataset
- target dataset
- category
- checkpoint path
- output directory
- start time and end time

Record losses when available:

- `loss_total`
- `loss_rec_clean`
- `loss_rec_aug_src`
- `loss_tpdr`
- `loss_tpdr_disp`
- `loss_tpdr_graph`
- `loss_tpdr_homeo`
- `loss_offset`

Record generator and deformation statistics when available:

- `mean_delta_p`
- `mean_delta_t`
- `mean_abs_A`
- `mean_abs_w`
- `mean_abs_Aw`
- `mean_Aw_violation`
- `token_interchange_ratio`
- `domain_code_norm`

Keep evaluation metric order:

```text
F-Score | CDL1 | CDL2 | EMDistance | UCD | UHD
```

Preferred machine-readable files:

- `metrics.jsonl` for step-level or epoch-level records.
- `summary.csv` for one row per experiment.
- TensorBoard logs for curves.

## Local Checks

When CUDA training is unavailable locally, run the strongest lightweight checks that fit the current environment.

Typical local checks:

- Python syntax compilation for edited files.
- Import checks when dependencies are available.
- Tiny CPU smoke tests for shape-only modules when possible.
- Config parsing checks.

Do not claim full training success unless the model actually trained or reached the requested validation point.

## Remote Training Rules

Remote server should be used for:

- full training
- validation and test
- CUDA extension compilation
- dataset-dependent runs

Each remote run should preserve:

- exact commit hash
- config file
- command line
- log file
- checkpoint path
- final and best metrics

Do not overwrite remote experiment directories. Use unique experiment names that include the method, dataset route, category, seed, and date or run id.

## Git Rules

Before substantial edits:

- Ensure this directory is a valid Git repository.
- Check for unrelated user changes.
- Do not revert unrelated changes.

After each completed module:

- Review the diff.
- Run available checks.
- Update the relevant checklist in `doc/tasks/`.
- Update `doc/tasks/progress.md` only when the task acceptance criteria are actually met.
- Commit with a clear message.

Recommended commit message style:

```text
stage1: clean dg backbone
stage1: add fixed sinpoint baseline
stage1: add dg training runner
stage2: add learnable msf generator
stage2: add tpdr loss
stage3: add dg deformable scan
```

If an experiment fails, do not hide it. Record the failure, commit hash, config, seed, symptom, and likely cause.

## Failure Handling

When a run fails or metrics regress:

1. Reproduce the failure from the same commit and config if possible.
2. Check whether the previous stable commit still works.
3. Disable the newest module with config switches.
4. Inspect `loss_total`, reconstruction losses, TPDR losses, offset losses, and deformation statistics.
5. Check for NaN, Inf, exploding deformation, or clamped offsets.
6. Change one variable at a time.
7. Record the result before trying the next change.

Do not stack multiple unverified fixes in one change.

## Review Checklist

The main agent should review every module diff for:

- Whether all prerequisite task files earlier in `doc/tasks/progress.md` are complete or explicitly waived.
- Whether it matches the assigned task file.
- Whether it changes files outside the expected scope.
- Whether it preserves tensor shapes.
- Whether disabled modules are true no-ops.
- Whether active training avoids target-domain training data.
- Whether losses match the design.
- Whether metric order and scaling are unchanged unless explicitly required.
- Whether logs are sufficient for later analysis.
- Whether local checks or remote smoke tests were run.

## Prompt Template For Module Agents

Use this template when starting a new module-development conversation:

```text
请先阅读 AGENTS.md、doc/proposal.md、doc/detailed-design.md 和 doc/tasks/<TASK>.md。
只完成 doc/tasks/<TASK>.md 中的任务，不要提前实现后续模块。
实现前先做 call-path audit，列出训练数据流、loss 数据流、model forward 返回值。
如果发现设计文档和现有工具函数语义不一致，先停下来说明，不要自行假设。
修改前先检查当前代码结构；修改后运行可用的轻量检查。
最后总结：改了哪些文件、满足了哪些验收标准、每个 checklist 项的文件和行号证据、哪些检查已运行、还有哪些风险。
```

## Prompt Template For Review

Use this template when asking the main agent to review:

```text
请作为 reviewer 审查当前修改，重点检查：
1. 按 doc/tasks/progress.md 顺序，前置任务是否已完成；如未完成，是否有明确豁免理由
2. 是否符合 doc/tasks/<TASK>.md
3. 是否破坏 Fixed SinPoint baseline 或已有稳定路径
4. tensor shape 是否一致
5. loss、metric、日志字段是否正确
6. 是否引入目标域训练、source-target alignment 或提前实现后续模块
7. 还需要哪些 smoke test 或远程实验
```
