# Paper Notes — Deep Residual Learning for Image Recognition

Reading notes on the paper this project reproduces. The goal is to be able to explain every design
decision and every claim. Numbers I have not yet checked against the PDF are marked **[verify]** and
will be confirmed in Phase 3.

**Citation.** K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.*
arXiv:1512.03385, Dec 2015. Published CVPR 2016. Microsoft Research.

**Headline.** Residual networks up to 152 layers won ILSVRC 2015 classification (3.57% top-5 error,
ensemble) and took first place in ImageNet detection/localization and COCO detection/segmentation.

---

## 1. The problem: the degradation problem

Empirical observation motivating the paper:

- Deeper plain networks (simple stacks of conv layers) are harder to optimize. Past some depth,
  **adding layers increases training error**, and test error with it.
- This is **not overfitting**. Overfitting = low training error, high test error. Here the training
  error itself is worse — the network cannot fit data it has already seen.
- It is **not a representational limitation**. There is a constructive proof that a deeper net can
  match a shallower one: keep the shallow net's layers, make the extra layers identity mappings. A
  solution with equal-or-lower training error provably exists; SGD just does not find it.
- The paper argues it is **unlikely to be vanishing gradients**. Their plain nets use Batch
  Normalization, which keeps forward-activation variances healthy; they also verified backward
  gradient norms are healthy. Neither forward nor backward signals vanish, yet optimization still
  fails. They conjecture "exponentially low convergence rates" for deep plain nets and explicitly
  leave the underlying cause open.

Takeaway for this project: the degradation problem is about **optimization difficulty /
conditioning**, not overfitting and not (simply) gradient magnitude. We must not summarize the paper
as "skip connections fix vanishing gradients" — the paper itself argues against that framing.

---

## 2. Residual learning

Let a few stacked layers be responsible for an underlying mapping `H(x)`. Rather than have them
learn `H(x)` directly, have them learn the **residual** `F(x) := H(x) - x`, and recover the target
as:

```
H(x) = F(x) + x
```

implemented with a **shortcut connection** that adds the block input to the block output.

**Hypothesis (the paper's, paraphrased).** If the optimal `H` is close to an identity mapping — and
their layer-response analysis suggests residual functions generally have small magnitude — then it
is easier for the solver to push `F` toward zero than to make a stack of nonlinear layers reproduce
an identity mapping. Referencing the problem to identity is a good preconditioning.

**Gradient view (complementary, to be measured, not asserted).** For `y = x + F(x)`,
`∂y/∂x = I + ∂F/∂x`. The identity term forwards gradient unchanged regardless of the residual
branch, so every block has a direct path to the loss. Later work (loss-landscape visualization;
"shattered gradients") adds that residual nets have smoother, better-conditioned loss surfaces. We
present this as supporting context and check it against our own gradient measurements in Exp 5.

---

## 3. Building blocks

**Basic block** (2 weighted layers) — used for CIFAR nets and ResNet-18/34:

```
x → [Conv3x3 → BN → ReLU → Conv3x3 → BN] → (+ x) → ReLU
```

- BN after each conv, before the addition.
- Second ReLU **after** the addition — the shortcut carries the clean signal into the sum and the
  nonlinearity acts on the combined result.
- A single-layer residual block (`y = W1 x + x`) showed no benefit in the paper — essentially linear,
  no advantage.

**Bottleneck block** (3 layers: 1×1 reduce → 3×3 → 1×1 restore) — used for ResNet-50/101/152 on
ImageNet. The 1×1 layers shrink then restore channel count so the 3×3 operates on fewer channels,
keeping compute manageable at high width. Parameter-free identity shortcuts matter more here: a
projection shortcut connects two high-dimensional ends and roughly doubles the block's cost.
**Not used in this project** — CIFAR nets use basic blocks only.

**Shortcut types.**
- **Identity**: input and output shapes match. Zero parameters, zero extra FLOPs.
- **Projection** `y = F(x) + Ws x`: `Ws` is a 1×1 conv used only to match shape when channel count
  or spatial size changes.

---

## 4. Architectures

### 4.1 ImageNet (reference, not reproduced)

Stem: 7×7 conv, 64, stride 2 → 3×3 max pool stride 2. Then four stages of blocks at 64/128/256/512
(basic) or with bottleneck expansion ×4. Downsampling by stride-2 conv at the first block of stages
2–4. Ends with global average pooling → 1000-way FC → softmax. Block counts per stage:
ResNet-18 `[2,2,2,2]`, ResNet-34 `[3,4,6,3]`, ResNet-50 `[3,4,6,3]` bottleneck, ResNet-101
`[3,4,23,3]`, ResNet-152 `[3,8,36,3]`.

### 4.2 CIFAR-10 (this is what we reproduce — paper Section 4.2)

- Input 32×32, per-pixel mean subtracted.
- First layer: 3×3 conv, 16 filters.
- Then `6n` layers: three stages on feature-map sizes 32 / 16 / 8, with `2n` layers each and
  16 / 32 / 64 filters. Downsampling by stride-2 convs at stage boundaries.
- Ends: global average pooling → 10-way FC → softmax.
- Total weighted layers: `6n + 2`.
- Shortcuts connect pairs of 3×3 layers (`3n` shortcuts). On CIFAR the paper uses **identity
  shortcuts in all cases** (option A, zero-padded for the channel increase), so **the residual model
  has exactly the same depth, width, parameter count, and FLOPs as the plain model**. This is the
  cleanest controlled comparison in the paper and the backbone of our project.

| n | 3 | 5 | 7 | 9 | 18 | 200 |
|---|---|---|---|---|----|-----|
| layers | 20 | 32 | 44 | 56 | 110 | 1202 |

---

## 5. Training recipe

### ImageNet (reference)
SGD, batch 256, momentum 0.9, weight decay 1e-4, LR 0.1 divided by 10 at plateaus, ~600k iterations,
He init, BN after every conv, no dropout, standard scale/color augmentation, 10-crop testing,
fully-convolutional multi-scale testing for best numbers.

### CIFAR-10 (reproduced)
- SGD, momentum 0.9, weight decay **1e-4**, He initialization, BN, **no dropout**.
- Batch size 128 (on 2 GPUs in the paper).
- LR **0.1**, ÷10 at **32k** and **48k** iterations, stop at **64k** iterations (≈182 epochs).
- **45k / 5k** train/val split.
- Augmentation: **4-pixel pad → random 32×32 crop → random horizontal flip**; per-pixel mean
  subtraction. **Test on the single original 32×32 view** (no TTA).
- ResNet-110: warm up at LR 0.01 until training error < ~80%, then return to 0.1. **[verify exact
  threshold wording]**

Our deviations (schedule length, normalization style, pad mode, hardware, AMP) are tabulated in
[project_specification.md](project_specification.md) §11.

---

## 6. Key results

### CIFAR-10 classification error (paper Table 6) — **[verify all against PDF]**

| Model | Layers | Params | Test error |
|-------|--------|--------|------------|
| ResNet-20 | 20 | 0.27M | 8.75% |
| ResNet-32 | 32 | 0.46M | 7.51% |
| ResNet-44 | 44 | 0.66M | 7.17% |
| ResNet-56 | 56 | 0.85M | 6.97% |
| ResNet-110 | 110 | 1.7M | ~6.4% (6.43 best; 6.61 ± 0.16 over 5 runs) |
| ResNet-1202 | 1202 | 19.4M | 7.93% |

Qualitative findings (confident; exact plain-net numbers **[verify]**):

- **Plain nets** (20/32/44/56): deeper → higher training and test error. Degradation is visible.
- **ResNets** (20/32/44/56): deeper → lower error, monotonically, in this range.
- **ResNet-1202** still reaches ~0.1% training error — no optimization failure even at 1202 layers —
  but generalizes worse than ResNet-110 (overfitting a small dataset with a huge model). This
  separates the *optimization* problem (solved by residual learning) from the *generalization*
  problem (not solved; needs regularization or more data).
- ResNet-110 needed LR warm-up for reliable convergence.
- **Layer responses** (Fig. 7): the std of layer outputs (post-BN, pre-addition) is smaller for
  ResNet layers than for the corresponding plain-net layers, and shrinks with depth for deeper
  ResNets — consistent with "residual functions are generally close to zero / optimal `H` is near
  identity".

### ImageNet (reference)

- Plain-34 has **higher training error** than plain-18 (degradation reproduced at ImageNet scale).
- ResNet-34 beats both ResNet-18 and plain-34; the gap between 18 and 34 *reverses sign* when
  shortcuts are added.
- Deeper ResNets (50/101/152) continue to improve. Single-model ResNet-152 ~4.49% top-5;
  ensemble 3.57% top-5 (ILSVRC 2015 winner). **[verify single-model numbers]**

### Identity vs projection shortcuts (paper Table 3, ImageNet)

Three options: (A) parameter-free zero-pad identity everywhere; (B) projection only where dimensions
change, identity elsewhere; (C) projection on all shortcuts. Result: `A < B < C` in accuracy but
**all differences small**; the paper attributes C's edge to added parameters and concludes
**projection shortcuts are not essential** for addressing degradation. They use B for ImageNet, A
for CIFAR (to keep plain/residual parameter-matched).

---

## 7. What the paper does and does not claim

**Does claim:**
- Residual learning eases *optimization* of very deep networks (the degradation problem).
- The effect is empirical and consistent across CIFAR and ImageNet, across depths, across tasks.
- Identity shortcuts are sufficient; projections are a convenience for shape matching.

**Does not claim:**
- That skip connections primarily fix vanishing gradients (argues against this).
- That deeper is always better for *generalization* (ResNet-1202 is the counterexample).
- A closed-form explanation of *why* plain nets are hard to optimize (left open).

---

## 8. Relevant follow-up work

- He et al., *Identity Mappings in Deep Residual Networks* (2016) — "pre-activation" ordering
  (BN→ReLU→Conv inside the block, clean identity path). Trains 1001-layer CIFAR nets. A natural
  extension experiment for us, not part of the core reproduction.
- "Bag of Tricks" (He et al., 2018) — `zero_init_residual`, cosine schedule, label smoothing, etc.
  We expose `zero_init_residual` as an off-by-default flag.
- Loss-landscape / shattered-gradient papers — theoretical support for the "smoother optimization"
  view; cited in `gradient_analysis.md`.

---

## 9. What this project takes from the paper

| Paper element | Our use |
|---------------|---------|
| Degradation problem | Exp 1 (reproduce on CIFAR at depths ≤ 56) |
| Residual reformulation | Exp 2 (parameter-matched plain vs residual) |
| Identity shortcuts on CIFAR | enables the controlled comparison; our default (option A) |
| Shortcut is the only difference | Exp 3 (remove it, expect plain-net behavior) |
| Identity vs projection ablation | Exp 4 (options A/B/C at depth 56) |
| "Not vanishing gradients" argument | Exp 5 (measure per-layer gradient norms, interpret carefully) |
| Layer-response analysis | Exp 5 supplement (residual-branch output magnitudes) |
| CIFAR training recipe | basis for our "standard" schedule (shortened to 80 epochs) |
| ResNet-1202 overfitting | discussion point: optimization ≠ generalization |

---

## 10. Numbers / details to verify against the PDF (Phase 3)

- [ ] CIFAR Table 6 exact error rates and the ResNet-110 mean ± std.
- [ ] Plain-net CIFAR error rates (Fig. 6) — currently only have the qualitative trend.
- [ ] ResNet-110 warm-up threshold exact wording ("training error below 80%").
- [ ] Exact CIFAR weight-decay and momentum (have 1e-4 / 0.9, confirm).
- [ ] Whether batch 128 is total or per-GPU (believe total).
- [ ] ImageNet single-model top-1/top-5 for ResNet-34/50/152.
- [ ] Exact wording of the vanishing-gradient argument in Section 4.1.
- [ ] Table 3 A/B/C numbers.
