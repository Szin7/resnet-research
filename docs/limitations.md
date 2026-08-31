# Limitations

_Status: living document. Constraints known from the design are listed now; experiment-specific
limitations are added as each experiment is run._

## By design (compute budget)

- **CIFAR-10, not ImageNet.** The paper's headline experiments are on ImageNet; we reproduce only
  the CIFAR-10 study (paper §4.2). Conclusions are about small 32×32 images and small models.
- **Depth ≤ 56 primary** (110 optional). We do not test the 1202-layer regime.
- **80-epoch schedule**, not the paper's ~182. Expect ~1–2 points below the paper's accuracy; the
  gap is reported, not hidden.
- **Single GPU (Colab T4)**, mixed precision for most runs.
- **Minor recipe deviations**: per-channel standardization vs per-pixel mean subtraction; reflect
  vs zero crop padding. Tabulated in [project_specification.md](project_specification.md) §11.

## Methodological

- Most experiments use **one seed** for the main comparison; run-to-run variation is quantified
  separately with seed replicates for resnet-20 and resnet-56. Gaps smaller than ~2× the measured
  std are reported as "within noise".
- The plain-vs-residual comparison is parameter-matched (option A) but **not** matched on everything
  one could imagine (e.g. effective learning dynamics differ because the shortcut changes the loss
  surface — that is the point).
- Removing BatchNorm forces a learning-rate change, so that ablation varies two things at once.
- Grad-CAM / feature-map visualizations (if included) show where activations are strong, not what
  the model "understands".

## Not claimed

State-of-the-art accuracy; exact numerical reproduction of the paper; that residual connections
"solve vanishing gradients"; production readiness.
