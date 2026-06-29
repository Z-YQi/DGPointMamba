# DGPointMamba Proposal

## 1. Project Summary

DGPointMamba is a source-only domain generalized point cloud completion project. It builds on the useful completion backbone components from DAPointMamba while removing source-target adaptation modules.

The goal is to train only on CRN / ShapeNet paired partial-complete point clouds and generalize to unseen target domains such as 3D-FUTURE, ModelNet, KITTI, ScanNet, and MatterPort3D.

Core idea:

> Learn source-only pseudo-domain geometric deformations from CRN partial point clouds, then use deformation-aware Mamba scanning to improve robustness to unseen target-domain partial inputs.

## 2. Motivation

Existing domain adaptive point cloud completion methods usually assume target-domain training data are available. This assumption is often unrealistic.

DAPointMamba is strong, but its key modules are designed for UDA:

- Cross-Domain Patch-Level Scanning.
- Cross-Domain Spatial SSM Alignment.
- Cross-Domain Channel SSM Alignment.

DGPointMamba removes those modules and studies a harder setting: source-only domain generalization.

The project should avoid directly transplanting 2D image DG or UDA mechanisms that depend on RGB appearance, image background, or adaptive image normalization. The target problem uses geometry-only point clouds, so the method should focus on deformation, local topology, patch scanning, and source-only pseudo-domain generation.

## 3. Proposed Contributions

1. **Source-only Mamba backbone for DG point cloud completion**
   - Keep DAPointMamba's useful completion backbone.
   - Remove active UDA source-target alignment.
   - Train without target-domain samples.
   - Support `model(partial)` for inference.

2. **Learnable MSF Domain Generator**
   - Extend SinPoint-MSF from random augmentation into a learnable source-only pseudo-domain generator.
   - Use multiple anchor-conditioned sine fields for local, topology-preserving deformation of source partial inputs.
   - In the main Scheme-A setting, deform only the source partial input and do not deform the complete supervision target.

3. **DGDeformableScan**
   - Apply domain-aware patch refining and deformation-aware offset prediction once before the 12-layer Mamba stack.
   - Use a standalone pre-Mamba module:

```text
domain-aware patch refining + offset prediction -> Mamba stack
```

   - Use clean / augmented branch token interchange to expose Mamba to source-only pseudo-domain variation.

4. **Topology-Preserving Deformation Regularization (TPDR)**
   - Regularize learnable pseudo-domain deformation using point displacement, local graph preservation, and homeomorphic deformation bounds.
   - Package the topology constraint as a core 3D geometry contribution rather than a generic augmentation penalty.

## 4. Scope

In scope:

- Source-only training loop.
- Backbone cleanup from DAPointMamba.
- Fixed SinPoint baseline.
- Global-parameter learnable SinPoint baseline.
- Learnable MSF Domain Generator.
- DomainSpecificEncoder for patch-level domain-specific geometric cues.
- DGDeformableScan.
- TPDR.
- Experiment configs and logging for DG evaluation.

Out of scope for the first implementation:

- Target-domain training data.
- Image-based adaptive normalization modules.
- Source-target adversarial alignment.
- Complete-target synchronous deformation in the main method.
- Full paper writing and final LaTeX tables.

## 5. Data and Evaluation Setting

Training:

- Source dataset: CRN / ShapeNet paired partial-complete point clouds.
- Training input: source partial point cloud and source complete point cloud.
- No target-domain training data.

Development ablation:

- Run ablations on CRN -> 3D-FUTURE first.
- Use 3D-FUTURE `CDL2` as the primary early-stage selection metric.

Full-model testing after the architecture is complete:

- 3D-FUTURE.
- ModelNet.
- KITTI.
- ScanNet.
- MatterPort3D.

Metrics:

- ModelNet and 3D-FUTURE: primary metric is `CDL2`.
- KITTI, ScanNet, and MatterPort3D: primary metrics are `UCD` and `UHD`.
- Keep output table format:

```text
F-Score | CDL1 | CDL2 | EMDistance | UCD | UHD
```

Metric scale:

- CDL1, CDL2, and UCD are multiplied by `10000` in the current code.
- UHD is multiplied by `100`.

## 6. High-Level Method

Training uses two source-only views:

1. Clean source view:
   - `P_s -> backbone -> pred_clean`.
   - Supervised by original complete point cloud `Y_s`.

2. Pseudo-domain augmented source view:
   - `P_s -> LearnableMSFDomainGenerator -> P_aug`.
   - `P_aug -> backbone -> pred_aug`.
   - Supervised by the original complete point cloud `Y_s`.

The clean and augmented branches share backbone weights.

This proposal uses **Scheme A** as the main training setting:

```text
P_s -> pred_clean, supervised by Y_s
P_aug = deform(P_s) -> pred_aug, supervised by Y_s
```

`Y_s` is never paired with target-domain data during training. A synchronously deformed complete supervision target is intentionally not used in the main method.

## 7. Tensor Shape Contract

Default symbols:

```text
B     = batch size
N     = 2048 partial points
M     = 2048 complete points
G     = 64 patches
S     = 32 points per patch
D     = 384 token dimension
D_dom = 64 domain-specific code dimension
k     = 4 MSF anchors
L     = mamba_depth
```

Core shapes:

```text
P_s:                [B, N, 3]
Y_s:                [B, M, 3]
P_aug:              [B, N, 3]
neighborhoods:      [B, G, S, 3]
centers:            [B, G, 3]
tokens:             [B, G, D]
domain_code_patch:  [B, G, D_dom]
domain_hint_patch:  [B, G, D_dom]
pos_embed:          [B, G, D]
Mamba output:       [B, G, D]
global feature:     [B, D, 1]
prediction:         [B, M, 3]
```

Every ablation must preserve the `[B, G, D]` token interface before entering the Mamba stack and the `[B, M, 3]` final prediction shape.

Canonical naming:

- `P_s`: source partial point cloud.
- `Y_s`: source complete point cloud.
- `P_aug`: pseudo-domain augmented source partial point cloud.
- `domain_hint_patch`: patch-level deformation descriptor produced by `LearnableMSFDomainGenerator`; used for logging or optional ablations, not required for inference.
- `domain_hint_global`: global deformation descriptor produced by `LearnableMSFDomainGenerator`.
- `domain_code_clean`: patch-level domain-specific code estimated by `DomainSpecificEncoder` for the clean branch.
- `domain_code_aug`: patch-level domain-specific code estimated by `DomainSpecificEncoder` for the augmented branch.
- `domain_code_patch`: generic patch-level domain-specific code name when branch distinction is unnecessary.
- `offsets`: concatenated `delta_p` and `delta_t`.

## 8. Development Stages

### Stage 1: Fixed SinPoint DG baseline

Goal: obtain the first reliable DG baseline using fixed SinPoint pseudo-domain augmentation.

- Backbone cleanup.
- Fixed SinPoint baseline.
- Global-parameter learnable SinPoint baseline.

Expected ablations:

```text
Fixed SinPoint
Global-parameter learnable SinPoint
```

Stage 1 does not require `DomainSpecificEncoder` or `DGDeformableScan`.

### Stage 2: Learnable MSF and TPDR

Goal: upgrade the generator from a global learnable sine perturbation to multi-anchor learnable pseudo-domain deformation.

- `LearnableMSFDomainGenerator`.
- Global learnable MSF parameters `[k, 3]` first.
- Optional future input-conditioned MSF parameters `[B, k, 3]`.
- `TPDR` as deformation regularization.

Expected ablations:

```text
Learnable MSF without TPDR
Learnable MSF with TPDR
```

### Stage 3: DGDeformableScan

Goal: inject source-only pseudo-domain geometry into the Mamba sequence modeling process.

- `DomainSpecificEncoder`.
- Standalone `DGDeformableScan` before the Mamba stack.
- Plain Mamba stack controlled by `mamba_depth`.
- Clean / augmented branch token interchange.
- Offset logging and offset regularization.

Expected ablations:

```text
Learnable MSF + TPDR
Learnable MSF + TPDR + DGDeformableScan
Full DGPointMamba
```

## 9. Expected Ablations

Early ablations are run on CRN -> 3D-FUTURE.

| Method | Fixed SinPoint | Learnable Global SinPoint | Learnable MSF | TPDR | DGDeformableScan | 3D-FUTURE CDL2 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed SinPoint baseline | yes | no | no | no | no | |
| Learnable global SinPoint | no | yes | no | optional | no | |
| Learnable MSF | no | no | yes | no | no | |
| Learnable MSF + TPDR | no | no | yes | yes | no | |
| Full DGPointMamba | no | no | yes | yes | yes | |

The augmented-branch reconstruction loss in all rows that use pseudo-domain augmentation is:

```text
L_rec_aug_src = CompletionLoss(pred_aug, Y_s)
```

## 10. Workflow

Recommended development workflow:

- Keep training environment and datasets on the remote server.
- Manage code with Git locally and remotely.
- Use Codex locally for code edits, configs, documentation, and experiment scripts.
- Use the remote server only for training, testing, and CUDA extension compilation.
- Track each experiment with commit hash, config, seed, dataset, checkpoint path, and metrics.

## 11. Success Criteria

The first successful milestone is:

- The fixed SinPoint DG baseline trains without target-domain training data.
- The model can run `model(partial)` for inference.
- Metrics are logged in a DGPointMamba experiment format.
- CRN -> 3D-FUTURE cabinet produces stable CDL2 logs.
- Fixed SinPoint and global-parameter learnable SinPoint can be compared with the same two-branch DG loss.
- Learnable SinPoint logs `A`, `w`, and related deformation magnitudes.

The full research milestone is:

- Learnable MSF + TPDR improves over fixed SinPoint and learnable global SinPoint baselines on 3D-FUTURE.
- DGDeformableScan further improves or stabilizes 3D-FUTURE generalization.
- The final full model is then evaluated on ModelNet, KITTI, ScanNet, and MatterPort3D.
