# Project Specification

Frozen reference for the ResNet-from-scratch reproduction. Everything here is fixed unless a change
is recorded in the "Change log" at the bottom with a date and reason. Discussion and background live
in [paper_notes.md](paper_notes.md); the mathematical treatment goes in `mathematics.md` (Phase 3);
this document is the "what", not the "why".

Status: Phase 2 draft, 2026-08-31. Nothing here has been executed yet. All performance numbers in
this document are targets or "to measure" — no measured results exist.

---

## 1. Scope

A lightweight from-scratch reproduction of the core residual-learning experiments from
He et al. (2015), *Deep Residual Learning for Image Recognition* (arXiv:1512.03385), Section 4.2
(CIFAR-10), under a Google Colab (T4) compute budget.

In scope: hand-implemented CIFAR ResNet / plain networks, parameter-matched comparisons, a depth
sweep, shortcut ablations, gradient-flow measurement, error analysis, computational benchmarking.

Out of scope: ImageNet, bottleneck blocks, networks deeper than 110 layers, distributed training,
hyperparameter search, deployment.

---

## 2. Research question and hypotheses (frozen)

**Primary:** How do residual (skip) connections affect the optimization and generalization of deep
convolutional networks on CIFAR-10, and does their benefit grow with depth?

| ID | Hypothesis | Falsification criterion | Fallback if not observed |
|----|------------|-------------------------|--------------------------|
| H1 | Plain-net test accuracy stops improving (and the deepest plain net shows higher *training* error than a shallower one) as depth goes 20→32→44→56. | Plain-56 training accuracy ≥ plain-20 training accuracy AND plain-56 test accuracy ≥ plain-44. | Extend depth to plain-74 / plain-110; report that degradation did not appear at 56 under this schedule. |
| H1' | ResNet test accuracy improves or holds flat across 20→56 with no training-error degradation. | ResNet-56 training accuracy < ResNet-20 training accuracy by > 1 point. | Report and investigate (likely under-training). |
| H2 | At depth ≥ 32 with identical parameter count, the residual model reaches a given training loss in fewer epochs and ends with higher test accuracy than the plain model; at depth 20 the gap is small (< 1 point). | Residual model is not better than plain at depth 44 or 56 on either training loss trajectory or test accuracy. | Report; check for a training bug in the plain path; consider longer schedule. |
| H3 | Removing only the additive shortcut from ResNet-56 (yielding an architecturally identical plain-56) increases final training loss and decreases test accuracy. | ResNet-56 and its shortcut-removed twin are within seed noise on both metrics. | Report as a null result; this would be a notable finding worth discussing. |
| H4 | Projection shortcuts (option B / C) land within ~1.0 point of identity shortcuts (option A); projections everywhere (C) add params/FLOPs without a reliable accuracy gain. | Option C beats option A by > 1.5 points at matched schedule. | Report; note it contradicts the paper's ImageNet finding. |
| H5 | Early in training, per-layer gradient-norm profiles are flatter across depth for ResNet-56 than for plain-56. | The two profiles are indistinguishable, or plain-56 is flatter. | Report; if plain-56 gradients are *not* vanishing, present this as support for the paper's "optimization, not gradient magnitude" position. |
| H6 | Ordering by test accuracy: deep-residual ≥ shallow-residual ≈ shallow-plain > deep-plain. | Any inversion larger than seed noise. | Report the actual ordering. |
| H7 | Without BatchNorm, LR 0.1 is unstable; a lower LR is needed, convergence is slower, final accuracy lower. | A no-BN model trains stably at LR 0.1 to within 1 point of its BN counterpart. | Report; note BN's benefit is smaller than expected at this depth. |
| H8 | A decaying LR schedule (step or cosine) beats a constant LR at the same initial value by a clear margin (> 2 points); cosine ≈ step at 80 epochs; momentum speeds early convergence. | Constant LR matches the decayed schedule within 1 point. | Report. |
| H9 | Standard augmentation (pad+crop+flip) narrows the train/test gap and improves test accuracy by ~1–3 points versus no augmentation. | No-augmentation test accuracy is within 0.5 points of augmented. | Report. |

Hypotheses are stated before running. Unexpected outcomes are documented, not suppressed.

---

## 3. Dataset specification (frozen)

| Property | Value |
|----------|-------|
| Dataset | CIFAR-10 (`torchvision.datasets.CIFAR10`, `download=True`) |
| Image size | 32 × 32 × 3, uint8 → float |
| Classes | 10, balanced (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Official split | 50,000 train / 10,000 test |
| Train/val split | 45,000 train / 5,000 val, random, **fixed with split seed 42** (identical across every run). Stratification not required (random split of a balanced set is ~balanced); log the realized per-class val counts. |
| Test set | The official 10,000. Untouched until final per-model evaluation. Never used for model selection. |
| Normalization | Per-channel standardization. Mean/std computed once from the **45k training split**, cached to `data/cifar10_stats.json`. Sanity-check targets: mean ≈ (0.4914, 0.4822, 0.4465), std ≈ (0.2470, 0.2435, 0.2616). |
| Deviation from paper | Paper uses per-pixel mean subtraction; we use per-channel standardization (standard modern practice). Recorded in §11. |

### Augmentation pipelines

| Name | Train transform | Eval transform (val + test) |
|------|-----------------|-----------------------------|
| `standard` (default) | Pad 4 (reflect) → RandomCrop 32 → RandomHorizontalFlip(p=0.5) → ToTensor → Normalize | ToTensor → Normalize |
| `minimal` (Exp 9 control) | ToTensor → Normalize | ToTensor → Normalize |

Padding mode: the paper zero-pads; we use reflect padding (marginally better, standard). Recorded as
a deviation. `standard` matches the paper's crop+flip otherwise. Test-time is always a single
un-augmented view (no test-time augmentation, no 10-crop).

---

## 4. Model specifications (frozen)

Shared rules: 3×3 convs have `padding=1`, `bias=False` (BN follows). No dropout in the ResNet/plain
family. ReLU is `inplace=True` except where it would break the residual add (applied to a fresh
tensor after the sum).

### 4.1 CIFAR ResNet / Plain family — `6n + 2` weighted layers

```
Input 3x32x32
  └─ conv1: Conv2d(3, 16, k3, s1, p1, bias=False) → BN(16) → ReLU        # 16 x 32 x 32
  └─ stage1: n BasicBlocks, 16 ch, stride 1                              # 16 x 32 x 32
  └─ stage2: BasicBlock(16→32, stride 2) + (n-1) BasicBlocks(32 ch)      # 32 x 16 x 16
  └─ stage3: BasicBlock(32→64, stride 2) + (n-1) BasicBlocks(64 ch)      # 64 x 8 x 8
  └─ head:  AdaptiveAvgPool2d(1) → Flatten → Linear(64, 10)             # 10
```

**BasicBlock** (2 weighted conv layers):

```
identity = x
out = Conv2d(in, out, k3, stride=s, p1, bias=False)(x)
out = BN(out)
out = ReLU(out)
out = Conv2d(out, out, k3, stride=1, p1, bias=False)(out)
out = BN(out)
if residual_enabled:
    out = out + shortcut(identity)      # shortcut per §4.2
out = ReLU(out)                          # note: after the add
return out
```

`residual_enabled=False` gives the **plain** network: identical layers, identical parameter count
(with shortcut option A), no additive skip. This flag is the single independent variable in Exp 2
and Exp 3.

### 4.2 Shortcut options

Applies when `stride != 1` or `in_channels != out_channels` (i.e., the first block of stage 2 and
stage 3). Within a stage, shortcuts are always plain identity.

| Option | Shortcut at dimension change | Extra params | Notes |
|--------|------------------------------|--------------|-------|
| **A** (default, paper's CIFAR choice) | `x[:, :, ::2, ::2]` then zero-pad channels to `out_channels` | 0 | Parameter-free. Plain and residual nets have *exactly* equal params. |
| **B** | `Conv2d(in, out, k1, stride=2, bias=False) → BN(out)`, only at the 2 stage entries | +2,752 (whole net) | Identity elsewhere. |
| **C** | Every shortcut is `Conv2d(in, out, k1, stride, bias=False) → BN(out)` | more (computed in code) | Includes within-stage 1×1 convs. |

### 4.3 Weight initialization (frozen)

| Layer | Init |
|-------|------|
| Conv2d | `kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')` |
| BatchNorm2d | weight = 1, bias = 0 |
| Linear (classifier) | PyTorch default (`kaiming_uniform_`), bias = 0 |
| `zero_init_residual` (flag, default **False**) | last BN weight in each BasicBlock set to 0. Post-paper trick; available for a micro-experiment, off for faithful runs. |

### 4.4 Expected parameter counts (test targets)

Computed from the architecture above, option A. These are asserted in `tests/test_resnet.py`.

| Model | n | Layers | Total params | Paper |
|-------|---|--------|--------------|-------|
| resnet-20 / plain-20 | 3 | 20 | **269,722** | 0.27M |
| resnet-32 / plain-32 | 5 | 32 | **464,154** | 0.46M |
| resnet-44 / plain-44 | 7 | 44 | **658,586** | 0.66M |
| resnet-56 / plain-56 | 9 | 56 | **853,018** | 0.85M |
| resnet-110 / plain-110 | 18 | 110 | **1,727,962** | 1.7M |

Option B adds 2,752 params (e.g. resnet-56-B = 855,770). Option C computed at test time.
Plain and residual (option A) are equal to the digit. Trainable == total (no frozen params).

### 4.5 Baseline CNNs (framing only, not a controlled comparison)

`SmallCNN` — a conventional pre-ResNet design, used to show the progression the way a first-time
reader expects, not as a parameter-matched control.

```
[Conv3x3→BN→ReLU→Conv3x3→BN→ReLU→MaxPool2] x3   with channels 32, 64, 128
Flatten → Linear(128*4*4, 256) → ReLU → Dropout(0.5) → Linear(256, 10)
```

`SmallCNN-deep` — same three-block skeleton with 4 conv layers per block (~12 conv layers), same
head. Deliberately keeps maxpool + FC + dropout (a different design philosophy from ResNet). Exact
param counts asserted in tests once implemented; both land near ~0.8–1.0M.

---

## 5. Training specification — "standard" schedule (frozen)

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Epochs | 80 | project choice (paper: ~182) |
| Batch size | 128 | paper |
| Optimizer | SGD | paper |
| Base learning rate | 0.1 | paper |
| Momentum | 0.9 | paper |
| Nesterov | False | paper |
| Weight decay | 1e-4 | paper |
| LR schedule (core) | step: ×0.1 at epoch 40 and 60 | mirrors paper's 50%/75% of total |
| Loss | `nn.CrossEntropyLoss` (mean reduction; includes log-softmax) | standard |
| Label smoothing | 0.0 | faithful |
| Gradient clipping | none | faithful |
| AMP (mixed precision) | **on** for §7 core sweep, Exp 4, Exp 8, Exp 9, variance runs; **off** for Exp 5 (gradient flow) and Exp 7 (batchnorm) | project choice — clean fp32 numbers where they matter |
| Warmup | none, except resnet-110 stretch runs: LR 0.01 until train acc ≥ 20%, then 0.1 | paper |
| Model selection | best checkpoint by **validation** accuracy | standard |
| Reported test accuracy | from best-val checkpoint AND from final-epoch checkpoint (both) | honesty |
| DataLoader | `num_workers=2`, `pin_memory=True`, `drop_last=False`, shuffle train only | Colab T4 |

Alternative schedules (`cosine`, `constant`) exist only as Exp 8 arms. `weight_decay=5e-4` is a
noted-but-untested alternative.

### Expected accuracy (targets, NOT results)

Community CIFAR ResNet reproductions at a comparable schedule land around: ResNet-20 ~90–91%,
ResNet-56 ~92–93% test accuracy — roughly 1–2 points below the paper's full-schedule numbers
(ResNet-20 91.25%, ResNet-56 93.03%). We expect similar and will verify. Any shortfall is attributed
to the 80-epoch schedule in the report.

---

## 6. Metrics — exact definitions (frozen)

| Metric | Definition |
|--------|------------|
| Top-1 accuracy | mean over the eval set of `argmax(logits) == label`. |
| Train loss (epoch) | running mean of per-batch cross-entropy over the epoch. |
| Val loss / acc | computed on the 5,000 val images each epoch, eval transform, `model.eval()`, no grad. |
| Test accuracy | on the 10,000 test images, once, at the end, from the selected checkpoint(s). |
| Per-class precision / recall / F1 | one-vs-rest from the 10×10 test confusion matrix. |
| Confusion matrix | rows = true label, cols = predicted; raw counts + a row-normalized copy for the figure. |
| Per-layer gradient norm | L2 norm of `weight.grad` for each Conv2d/Linear, captured after `backward()`, before `optimizer.step()`, on a fixed batch (fixed seed, AMP off). Also aggregate: mean / median / max / min across layers. |
| Epochs-to-threshold | first epoch with train accuracy ≥ {60%, 80%} — convergence-speed proxy. |
| Parameter count | `sum(p.numel() for p in model.parameters())`; trainable counted separately. |
| Model size on disk | byte size of the saved `state_dict` (fp32). |
| MACs | multiply-accumulates for one forward pass on a single 3×32×32 input, measured with a counting tool (record which: `fvcore` or `ptflops`). |
| Inference latency | mean ± std wall-clock of a no-grad forward pass, batch size 1 and 128, after 10 warmup iters, 50 timed iters; device recorded. |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` during one training step at batch 128. |
| Training time | wall-clock per epoch and total, logged per run, device recorded. |

---

## 7. Core training sweep (feeds Experiments 1, 2, 3, 6)

Eight runs, standard schedule, seed 1, shortcut option A, BN on, `standard` augmentation. Trained
once, reused across experiments.

| Run | model.name | depth | residual |
|-----|-----------|-------|----------|
| plain-20 | plain | 20 | — |
| plain-32 | plain | 32 | — |
| plain-44 | plain | 44 | — |
| plain-56 | plain | 56 | — |
| resnet-20 | resnet | 20 | A |
| resnet-32 | resnet | 32 | A |
| resnet-44 | resnet | 44 | A |
| resnet-56 | resnet | 56 | A |

Plus `SmallCNN` and `SmallCNN-deep` (framing).

---

## 8. Experiment protocols

Every experiment write-up (`experiments/NN_name/README.md`) uses this structure:
**Research question → Hypothesis → Setup (fixed / varied) → Results (measured only) → Interpretation
→ Limitations.**

| # | Name | New runs | Fixed | Varied | Deliverables |
|---|------|----------|-------|--------|--------------|
| 1 | Depth | 0 (core sweep) | recipe, seed, data | depth ∈ {20,32,44,56}, family ∈ {plain, resnet} | train-error-vs-depth plot, test-acc-vs-depth plot, table |
| 2 | Plain vs residual | 0 (core sweep) | depth, params, recipe | residual on/off | loss curves (train + val) overlaid by family, convergence table |
| 3 | Shortcut ablation | 1 confirm (resnet-56 with skips zeroed at runtime, to double-check it equals plain-56) | architecture, params, recipe | additive skip only | Δ train loss, Δ test acc, side-by-side curves |
| 4 | Identity vs projection | 2 (resnet-56-B, resnet-56-C) | depth, recipe | shortcut ∈ {A, B, C} | params/MACs/acc/time table, tensor-shape walkthrough at both transitions |
| 5 | Gradient flow | 2 short instrumented (plain-56, resnet-56; ~5 epochs, AMP off, grad logging on) | recipe, data, fixed probe batch | family | per-layer grad-norm profile (at init, epoch 1, epoch 5), aggregate-vs-depth plot |
| 6 | Ablation grid | 0–1 (maybe plain-56-noBN) | recipe | residual × depth × BN | structured table: config → test acc, train acc, epochs-to-80% |
| 7 | BatchNorm | 2 (resnet-20-noBN, plain-20-noBN) + LR retune attempts (AMP off) | depth, data | BN on/off, and the LR change it forces | max stable LR probe, convergence curves, final-acc table |
| 8 | Optimization | 3 (resnet-20: constant LR, cosine, no-momentum) | model, data | LR schedule ∈ {step, cosine, constant}, momentum ∈ {0.9, 0.0} | loss curves, final-acc table, epochs-to-threshold |
| 9 | Augmentation | 1 (resnet-20 + `minimal`) | model, recipe | augmentation ∈ {standard, minimal} | train/test gap table, curves |
| 10 | Error analysis | 0 (evaluate best checkpoint) | — | — | confusion matrix figure, per-class table, misclassification gallery (≥ 24 examples), `docs/error_analysis.md` |

Supplementary (no training): variance runs (resnet-20 × seeds {2,3}, resnet-56 × seeds {2,3}) →
mean ± std; benchmarking table for every model; optional interpretability (feature maps, Grad-CAM
with caveats).

---

## 9. Configuration schema

Runs are composed from small YAML fragments. Full resolved config is dumped into each run directory.

```yaml
seed: 1
split_seed: 42          # fixed forever; controls train/val split only

model:
  name: resnet          # resnet | plain | small_cnn | small_cnn_deep
  depth: 56             # -> n = (depth - 2) / 6, must be integer for resnet/plain
  shortcut: A           # A | B | C   (resnet only)
  use_bn: true
  zero_init_residual: false

data:
  dataset: cifar10
  val_size: 5000
  augment: standard     # standard | minimal
  normalize: per_channel

training:
  epochs: 80
  batch_size: 128
  optimizer: sgd
  lr: 0.1
  momentum: 0.9
  nesterov: false
  weight_decay: 0.0001
  scheduler: step       # step | cosine | constant
  step_milestones: [40, 60]
  step_gamma: 0.1
  warmup: none          # none | resnet110
  amp: true

logging:
  out_dir: results/runs
  run_name: resnet-56_seed1     # dir name; auto-generated if omitted
  save_best: true
  save_final: true
  grad_flow: false      # per-layer gradient-norm logging
  grad_flow_batches: [0]  # which global steps to probe when grad_flow on
```

Validation at load time: `(depth - 2) % 6 == 0` for resnet/plain; `shortcut` only with those;
`scheduler == step` requires `step_milestones`; `amp` forced off if `grad_flow` true or `use_bn`
false (with a logged warning).

An experiment file lists runs as `base + overrides` and names the figures/tables to regenerate:

```yaml
# experiments/08_optimization/experiment.yaml
base: configs/models/resnet_20.yaml
runs:
  - {name: step,        overrides: {training.scheduler: step}}
  - {name: cosine,      overrides: {training.scheduler: cosine}}
  - {name: constant,    overrides: {training.scheduler: constant}}
  - {name: no_momentum, overrides: {training.momentum: 0.0}}
figures: [loss_curves, final_acc_bar]
```

---

## 10. Reproducibility manifest

Logged per run into `results/runs/<run_name>/`:

| Artifact | Contents |
|----------|----------|
| `config.yaml` | fully resolved config |
| `env.json` | Python, torch, torchvision, CUDA, cuDNN versions; GPU name; platform; `pip freeze` hash; git commit (once repo is versioned) |
| `metrics.csv` | per epoch: train_loss, train_acc, val_loss, val_acc, lr, epoch_seconds |
| `summary.json` | best epoch, best val acc, test acc (best-val + final), per-class metrics, total time, peak GPU mem, param count |
| `grad_flow.json` | (if enabled) per-layer gradient norms at the probed steps |
| `checkpoints/best.pt`, `checkpoints/final.pt` | `state_dict` + epoch + val acc + optimizer state (final only) |

Determinism policy: set `seed` for Python / NumPy / torch (CPU + CUDA); `cudnn.deterministic=True`,
`cudnn.benchmark=False`; `torch.use_deterministic_algorithms(True, warn_only=True)`. Honest claim:
**"seeded and closely reproducible; small run-to-run variation is possible on GPU."** Quantified by
the variance runs — the test-accuracy std across seeds {1,2,3} is reported alongside every headline
comparison, and any claimed gap smaller than ~2× that std is called "within noise".

---

## 11. Deviations from the paper (frozen table)

| Aspect | Paper | This project | Why |
|--------|-------|--------------|-----|
| Dataset | ImageNet + CIFAR-10 | CIFAR-10 only | compute |
| Depths | 20–110, plus 1202 | 20–56 primary; 110 optional | compute / time across ~22 runs |
| Schedule | 64k iters (~182 epochs), step ×0.1 at 32k/48k | 80 epochs, step ×0.1 at 40/60 | wall-clock budget |
| Normalization | per-pixel mean subtraction | per-channel standardization | modern standard practice |
| Crop padding | zero pad 4 | reflect pad 4 | small, standard |
| Hardware | 2 GPUs | 1 × Colab T4 | stated constraint |
| Mixed precision | none (fp32) | AMP on for most runs, off where numbers must be clean | T4 throughput |
| Expected ResNet-20 test error | 8.75% | ~9–11% (to measure) | shorter schedule |

No claim of numerical parity is made anywhere unless a faithful-schedule run achieves it.

---

## 12. Compute & storage budget (Colab T4)

Per-run estimates (80 epochs, AMP on where applicable) — **planning estimates, to be replaced by
measured `epoch_seconds`**:

| Model | ~per run |
|-------|----------|
| resnet/plain-20, SmallCNN | ~12–16 min |
| resnet/plain-32 | ~16–22 min |
| resnet/plain-44 | ~22–30 min |
| resnet/plain-56 | ~28–40 min |
| resnet-110 (stretch) | ~55–80 min |

| Bucket | Runs | ~Compute |
|--------|------|----------|
| Core sweep (§7) + 2 baselines | 10 | ~3.5 h |
| Exp 4 | 2 | ~1.2 h |
| Exp 5 (short) | 2 | ~15 min |
| Exp 7 (+ retune) | 2–4 | ~1 h |
| Exp 8 | 3 | ~0.7 h |
| Exp 9 | 1 | ~0.25 h |
| Variance | 4 | ~1.5 h |
| Re-runs / failures margin | — | ~2 h |
| **Total** | ~24 | **~11–13 h → 2–4 Colab sessions** |

Storage: CIFAR-10 ~180 MB; checkpoints ~2–7 MB each, best+final per run → ~150–250 MB; metrics +
figures + reports < 50 MB. **Total footprint ~0.5–0.7 GB**, well under the 5 GB ceiling. `data/` and
`checkpoints/` are gitignored; `results/runs/*/metrics.csv` + `summary.json` + `results/figures/` are
committed.

Colab workflow: mount Google Drive; CIFAR-10 and all `results/runs/` written to Drive so a dropped
session loses nothing; training loop checkpoints every epoch and resumes from `final.pt` if present.

---

## 13. Change log

| Date | Change | Reason |
|------|--------|--------|
| 2026-08-31 | Initial spec (Phase 2). | — |
