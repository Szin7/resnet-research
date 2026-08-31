# Mathematical Foundations

The math needed to understand and defend this project. Each section states the operation, its
gradient (since backpropagation is just the chain rule applied through these), and why it is used
here. Notation is kept consistent throughout.

Companion docs: [paper_notes.md](paper_notes.md) (what the paper does),
[project_specification.md](project_specification.md) (what we freeze). The gradient-flow section here
is expanded with measurements in `gradient_analysis.md` (Phase 14).

---

## 0. Notation

| Symbol | Meaning |
|--------|---------|
| $x$ | input to a layer or block (a tensor; often $\mathbb{R}^{C\times H\times W}$ for one image) |
| $C_{in}, C_{out}$ | input / output channel counts |
| $H, W$ | spatial height, width of a feature map |
| $k$ | convolution kernel size (we use $k=3$, sometimes $k=1$) |
| $s$ | stride; $P$ | zero/reflect padding width |
| $W, b$ | learnable weights, biases |
| $\theta$ | all trainable parameters of the network |
| $\eta$ | learning rate; $\mu$ | momentum coefficient; $\lambda$ | weight-decay coefficient |
| $\mathcal{L}$ | scalar loss |
| $z$ | logits (pre-softmax network outputs), $z\in\mathbb{R}^K$, $K=10$ |
| $p$ | predicted class probabilities; $t$ | one-hot target |
| $\sigma(\cdot)$ | ReLU, unless stated otherwise |
| $*$ | convolution (cross-correlation, see §1) |

Backpropagation, in one sentence: run the forward pass and cache activations; then sweep backward,
and at each operation use $\dfrac{\partial \mathcal{L}}{\partial(\text{output})}$ to produce both
$\dfrac{\partial \mathcal{L}}{\partial(\text{input})}$ (to pass further back) and
$\dfrac{\partial \mathcal{L}}{\partial(\text{params})}$ (to update). The cost is about twice a
forward pass regardless of parameter count. The per-operation rules are given below.

---

## 1. Convolution

### Operation

A convolutional layer has $C_{out}$ filters, each a weight block of shape $C_{in}\times k\times k$.
Collected: $W\in\mathbb{R}^{C_{out}\times C_{in}\times k\times k}$, plus optional bias
$b\in\mathbb{R}^{C_{out}}$ (we set `bias=False` whenever BatchNorm follows, since BN's $\beta$ absorbs
it). The output feature map is

$$
Y[o,i,j] \;=\; b[o] \;+\; \sum_{c=0}^{C_{in}-1}\sum_{p=0}^{k-1}\sum_{q=0}^{k-1}
W[o,c,p,q]\; X\big[c,\; s\,i + p - P,\; s\,j + q - P\big].
$$

Deep-learning "convolution" is really cross-correlation (no kernel flip); since $W$ is learned, the
distinction does not matter.

**Kernels / filters.** Each filter is a small pattern detector slid across every spatial position.
The same weights are reused at all positions (weight sharing), which gives approximate translation
equivariance and far fewer parameters than a dense layer over the same input.

**Channels.** Every output channel $o$ is a sum over *all* input channels — one filter looks at the
whole depth of the input. $C_{out}$ = number of filters = number of distinct features this layer
extracts. In our CIFAR ResNet $C$ goes $3 \to 16 \to 32 \to 64$.

**Stride $s$.** The step of the slide. $s=1$ keeps resolution; $s=2$ halves it and is how we
downsample (no pooling layers in the ResNet family here).

**Padding $P$.** Border pixels added before convolving. With odd $k$, $P=(k-1)/2$, $s=1$ keeps the
spatial size ("same" padding); it also stops the borders from being under-sampled. We use $P=1$ for
$3\times3$ convs. The paper zero-pads; we reflect-pad the *input augmentation* crop but the conv
layers themselves use zero padding.

**Feature map.** One output channel — a 2-D grid of one filter's responses. The layer output is the
stack of all $C_{out}$ feature maps.

**Output size.**

$$
H_{out} \;=\; \left\lfloor \frac{H_{in} + 2P - k}{s} \right\rfloor + 1 .
$$

Worked: $3\times3$, $P=1$, $s=1$, $H_{in}=32 \Rightarrow 32$. With $s=2$: $32\to16\to8$ over the
three stages, matching the architecture.

**Receptive field.** The input region influencing one output unit. It grows with depth; two stacked
$3\times3$ convs see a $5\times5$ region with fewer parameters and an extra nonlinearity compared to
one $5\times5$ conv — the reason modern CNNs favour stacks of small kernels.

### Gradient

A conv layer is linear in both $X$ and $W$, so its backward rules are themselves convolution-like:

- $\partial\mathcal{L}/\partial X$ is a *transposed* convolution of $\partial\mathcal{L}/\partial Y$
  with $W$ (stride becomes upsampling).
- $\partial\mathcal{L}/\partial W$ is a convolution of the input $X$ with $\partial\mathcal{L}/\partial Y$.
- $\partial\mathcal{L}/\partial b[o] = \sum_{i,j}\partial\mathcal{L}/\partial Y[o,i,j]$.

### Cost

Parameters: $C_{out}\,C_{in}\,k^2$ (no bias). Multiply-accumulates for one image:
$C_{out}\,H_{out}\,W_{out}\,C_{in}\,k^2$. (Exact per-model MAC counts are measured with a tool in
Phase 18, not asserted here.)

**Worked parameter count — ResNet-20, shortcut option A** (cross-checks the target in the spec):

| Part | Arithmetic | Params |
|------|-----------|-------:|
| conv1 | $3\cdot3\cdot3\cdot16$ + BN $2\cdot16$ | 464 |
| stage 1 | $3$ blocks $\times\,2\,(3\cdot3\cdot16\cdot16 + 2\cdot16)$ | 14,016 |
| stage 2 | first block $(16\!\to\!32,\,s2)$ 13,952 + $2\times$ 18,560 | 51,072 |
| stage 3 | first block $(32\!\to\!64,\,s2)$ 55,552 + $2\times$ 73,984 | 203,520 |
| head | FC $64\cdot10 + 10$ | 650 |
| **total** | | **269,722** |

Option A shortcuts add zero parameters, so plain-20 has the identical count — the basis of the
controlled comparison.

---

## 2. Activation — ReLU

$$
\sigma(x) = \max(0, x), \qquad
\sigma'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}
$$

(undefined at $0$; PyTorch uses $0$). Backward rule: $\partial\mathcal{L}/\partial x =
\partial\mathcal{L}/\partial y \cdot \mathbb{1}[x>0]$ — the incoming gradient is passed through
unchanged where the pre-activation was positive, and blocked where it was negative.

**Why ReLU here.**

- Cheap (a threshold) and non-saturating for $x>0$: the derivative there is exactly $1$, so it does
  not attenuate the backward signal. Sigmoid/tanh have derivative $\le 0.25$ and $\to 0$ in the
  tails; chaining many such layers shrinks gradients geometrically. ReLU was a key enabler of
  training networks deeper than a few layers.
- Produces sparse activations (hard zeros for negative pre-activations).

**Limitations.** A unit whose pre-activation is always negative receives zero gradient forever
("dying ReLU"). BatchNorm and He initialization keep pre-activations well-centred, so this is rarely
an issue at our scale. Variants (LeakyReLU, ELU, GELU) address it; the paper and this project use
plain ReLU.

**Caveat for the depth story.** ReLU alone does not guarantee healthy gradients through a deep
stack — the product of many weight Jacobians can still vanish or explode depending on their scale.
That is the job of initialization (§8.0) and BatchNorm (§3), and ultimately of the residual
structure (§4–5).

---

## 3. Batch Normalization

### Purpose

Normalize the distribution of each layer's inputs during training so that the optimization problem
is better conditioned. Ioffe & Szegedy (2015) motivated it as reducing *internal covariate shift*
(layers chasing a moving input distribution). Later analysis (Santurkar et al., 2018) argues the
practical mechanism is a **smoother loss landscape** — gradients become more Lipschitz and more
predictive of the loss a step ahead. Both framings agree on the observable effects.

### Formulation

For a channel, over a mini-batch of $m$ examples (and, for conv layers, all spatial positions, so
$m\cdot H\cdot W$ values):

$$
\mu_\mathcal{B} = \frac{1}{m}\sum_{i} x_i, \qquad
\sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i}(x_i - \mu_\mathcal{B})^2,
$$

$$
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \qquad
y_i = \gamma\,\hat{x}_i + \beta .
$$

$\gamma$ (scale) and $\beta$ (shift) are learnable, one pair per channel — so a conv layer with $C$
output channels adds $2C$ parameters (these are the "+ BN" terms in the §1 count). $\epsilon$
(PyTorch default $10^{-5}$) guards the division.

**Train vs. eval.** Training uses the batch statistics above and updates running estimates by
exponential moving average, $\hat\mu \leftarrow (1-\rho)\hat\mu + \rho\,\mu_\mathcal{B}$ (PyTorch
`momentum` $\rho = 0.1$). Evaluation uses the fixed running $\hat\mu, \hat\sigma^2$, so the output is
a deterministic function of a single input — important for reproducible test numbers and for the
gradient probes in Exp 5.

### Effects relevant to this project

- **Enables the LR of 0.1.** Both plain and residual nets in the paper depend on BN to train at this
  rate; §8 (Exp 8) and Exp 7 probe this.
- **Scale-invariance to preceding weights.** For $\alpha>0$, ignoring $\epsilon$,

$$
\mathrm{BN}(\alpha W x) = \mathrm{BN}(W x),
$$

  because $\mu$ and $\sigma$ scale with $\alpha$ too. The magnitude of a BN-preceded weight matrix
  does not change the function computed — it only changes the *effective* step size for that layer.
  This is why weight decay behaves unusually in BN networks (§8.4).
- **Mild regularization** from the noise in batch statistics.
- **What it does *not* do.** The paper trains *plain* nets with BN and still sees the degradation
  problem, and verifies their gradient norms are healthy. So BN fixes the raw signal-scale problem;
  the residual-specific benefit (§4–5) is something else. Exp 7 removes BN to measure what it buys.

### Gradient

BN is differentiable. $\partial\mathcal{L}/\partial\gamma = \sum_i (\partial\mathcal{L}/\partial y_i)\,\hat{x}_i$,
$\partial\mathcal{L}/\partial\beta = \sum_i \partial\mathcal{L}/\partial y_i$. The input gradient
$\partial\mathcal{L}/\partial x_i$ has three terms (a direct path, a path through $\mu_\mathcal{B}$,
a path through $\sigma_\mathcal{B}^2$) and, crucially, *centres and rescales the backward signal*:
the mean and the $\hat{x}$-projection of the incoming gradient are subtracted off. This is the
mechanism behind "smoother gradients".

---

## 4. Residual Learning

### The reformulation

Let $x$ be a block's input and $\mathcal{H}(x)$ the mapping we want the block to realize. Instead of
parameterizing $\mathcal{H}$ directly, parameterize the **residual**

$$
\mathcal{F}(x) := \mathcal{H}(x) - x,
$$

and reconstruct the target by adding the input back through a shortcut connection:

$$
y = \mathcal{F}(x, \{W_i\}) + x, \qquad \text{block output} = \sigma(y).
$$

For our BasicBlock, $\mathcal{F}(x) = \mathrm{BN}\big(W_2 * \sigma(\mathrm{BN}(W_1 * x))\big)$ (biases
folded into BN).

**Shape changes.** $\mathcal{F}(x)$ and $x$ must have the same shape to be added. When a stage
changes channel count and halves resolution (stage 2 and stage 3 entry blocks), use a projection:

$$
y = \mathcal{F}(x, \{W_i\}) + W_s\,x,
$$

where $W_s$ is a $1\times1$ conv with the matching stride (option B/C). Option A instead replaces
$W_s x$ with a parameter-free operation: subsample $x$ by taking every other pixel and zero-pad the
new channels. Tensor shapes at the stage-2 entry, option A:

```
x:            [N, 16, 32, 32]
F(x):         [N, 32, 16, 16]    (conv stride 2, 16->32)
subsample x:  [N, 16, 16, 16]    (x[:, :, ::2, ::2])
zero-pad ch:  [N, 32, 16, 16]    (cat 16 zero channels)
y = F(x) + shortcut(x):  [N, 32, 16, 16]
```

### Why reformulate — three arguments

1. **Identity becomes the cheap default.** If the best $\mathcal{H}$ for a block is near identity,
   the solver only needs $\mathcal{F}\to 0$ (drive the conv weights small), which is far easier than
   making a conv–BN–ReLU–conv–BN stack reproduce its own input.
2. **The "no worse than shallower" solution becomes reachable.** A deeper net can always emulate a
   shallower one by making extra blocks identity. With plain blocks that requires *learning*
   identity; with residual blocks it is just $\mathcal{F}=0$. Residual nets remove the excuse for a
   deeper model to have higher training error.
3. **It matches the structure of good solutions.** The paper's Fig. 7 shows learned residual
   branches have small output magnitude — solutions do sit near the identity reference.

### What residual learning is *not*

- Not a claim that every block *is* near-identity — only that referencing to identity conditions the
  search.
- Not a fix for generalization: ResNet-1202 optimizes fine and still overfits CIFAR-10.
- Not primarily a vanishing-gradient fix (§5).

---

## 5. Gradient Flow through the shortcut

### The additive path

Take the clean recurrence (this is exact for the *pre-activation* block of He et al. 2016; for the
original block, where a ReLU sits on $y$, it holds approximately because $\sigma'\in\{0,1\}$):

$$
x_{\ell+1} = x_\ell + \mathcal{F}_\ell(x_\ell, W_\ell).
$$

Unrolling from block $\ell$ to a deeper block $L$:

$$
x_L = x_\ell + \sum_{i=\ell}^{L-1}\mathcal{F}_i(x_i, W_i).
$$

Differentiate the loss w.r.t. the earlier activation:

$$
\boxed{\;
\frac{\partial\mathcal{L}}{\partial x_\ell}
= \frac{\partial\mathcal{L}}{\partial x_L}\left(1 + \frac{\partial}{\partial x_\ell}
\sum_{i=\ell}^{L-1}\mathcal{F}_i(x_i, W_i)\right)
\;}
$$

The leading $1$ means $\partial\mathcal{L}/\partial x_L$ reaches block $\ell$ **directly and
undistorted**, no matter what the intervening weights are. The gradient at $x_\ell$ can only vanish
if the second term equals exactly $-1$ — which will not happen consistently across a mini-batch.

### Contrast with a plain network

$x_{\ell+1} = \mathcal{F}_\ell(x_\ell, W_\ell)$ gives a **product** of Jacobians:

$$
\frac{\partial\mathcal{L}}{\partial x_\ell}
= \frac{\partial\mathcal{L}}{\partial x_L}\prod_{i=\ell}^{L-1}\frac{\partial\mathcal{F}_i}{\partial x_i}.
$$

If those Jacobians have spectral norm consistently below $1$, the gradient shrinks geometrically in
depth; above $1$, it explodes. The residual form turns this multiplicative process into an additive
one (identity plus a sum).

### The careful version (this is the part not to oversimplify)

1. **With BatchNorm and He init, plain-net Jacobians are roughly norm-preserving.** The paper
   measured healthy gradient norms in its plain nets. So the additive-vs-multiplicative argument
   explains why *very deep* residual nets stay trainable (1202 layers), but it is **not** by itself
   the explanation for the degradation problem at moderate depth. That degradation is better
   described as an optimization-landscape / conditioning problem whose exact cause the paper leaves
   open.
2. **The original block's trailing ReLU breaks the exact identity path.** The boxed identity is for
   pre-activation ResNets; the original ResNet still works, so the precise algebra is not the whole
   story.
3. **"Better gradient flow" is a hypothesis we test**, not assume. Exp 5 measures per-layer gradient
   norms in plain-56 vs. resnet-56. The likely outcome, consistent with the paper, is that plain-56
   gradients are *not* vanishing — in which case that itself is the interesting result.

What can be stated safely: the identity path guarantees a floor on how much gradient signal reaches
early layers, empirically smooths the loss surface (Li et al., 2018; Balduzzi et al., 2017), and
removes the requirement that every nonlinear layer preserve information for the layers behind it.

---

## 6. Softmax

The network outputs logits $z\in\mathbb{R}^K$. Softmax turns them into a probability distribution:

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad p_k\in(0,1),\quad \sum_k p_k = 1.
$$

**Shift-invariance.** $\mathrm{softmax}(z + c\mathbf{1}) = \mathrm{softmax}(z)$: only differences of
logits matter. Implementations subtract $\max_j z_j$ before exponentiating for numerical stability.

**Temperature.** $\mathrm{softmax}(z/T)$: $T\to 0$ approaches a one-hot argmax, $T\to\infty$
approaches uniform. (Used when discussing calibration / Grad-CAM.)

---

## 7. Cross-Entropy loss

For target class $y$ (one-hot $t$), with predicted distribution $p$:

$$
\mathcal{L} = -\sum_{k=1}^{K} t_k \log p_k = -\log p_y .
$$

Over a batch of $N$: $\mathcal{L} = -\frac{1}{N}\sum_{n} \log p_{y_n}$ (mean reduction, per the spec).

**Interpretation.** This is the negative log-likelihood of the correct labels under the model;
minimizing it is maximum-likelihood. Since $\mathcal{L} = H(t,p) = H(t) + D_{\mathrm{KL}}(t\,\|\,p)$
and $H(t)=0$ for a one-hot target, minimizing cross-entropy minimizes the KL divergence from the
prediction to the target.

**The fused gradient.** Composing softmax and cross-entropy, the gradient w.r.t. the logits is just

$$
\frac{\partial\mathcal{L}}{\partial z_k} = p_k - t_k .
$$

Derivation: $\mathcal{L} = -z_y + \log\sum_j e^{z_j}$, so
$\partial\mathcal{L}/\partial z_k = -\mathbb{1}[k=y] + p_k$. This is why `nn.CrossEntropyLoss` takes
raw logits (not probabilities): the combined backward pass is "prediction minus target" — stable and
$O(K)$. Weight decay is applied by the optimizer, not added to $\mathcal{L}$.

**Label smoothing** (kept at $0$ in this project): replace $t$ with $(1-\alpha)t + \alpha/K$.

---

## 8. Optimization

### 8.0 Initialization (He / MSRA)

Before optimization runs, weights are set so that activation and gradient variance are preserved
across layers. For a conv/linear layer feeding a ReLU, draw

$$
W \sim \mathcal{N}\!\left(0,\; \frac{2}{n}\right)
$$

(He et al., 2015, *Delving Deep into Rectifiers*), where $n$ is the fan-in $C_{in}k^2$ (preserves
activation variance on the forward pass) or the fan-out $C_{out}k^2$ (preserves gradient variance on
the backward pass). The factor $2$ compensates for ReLU zeroing half the pre-activations in
expectation. Following common ResNet practice we use `mode='fan_out'`. BN weights start at $1$,
biases at $0$; conv layers have no bias.

### 8.1 Gradient descent

Objective (with L2 regularization, §8.4):

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{n=1}^{N}\ell_n(\theta) + \frac{\lambda}{2}\lVert\theta\rVert_2^2 .
$$

Full-batch update: $\theta \leftarrow \theta - \eta\,\nabla_\theta\mathcal{L}(\theta)$. Computing the
exact gradient over all $N$ every step is wasteful.

### 8.2 Stochastic gradient descent (mini-batch)

Estimate the gradient from a random mini-batch $\mathcal{B}$, $|\mathcal{B}| = m = 128$:

$$
\hat{g} = \frac{1}{m}\sum_{n\in\mathcal{B}}\nabla_\theta \ell_n(\theta), \qquad
\theta \leftarrow \theta - \eta\,\hat{g}.
$$

$\hat g$ is an unbiased estimate of the full gradient. Its noise is not purely harmful: it helps
escape saddle points and sharp minima and is linked to better generalization. Batch size trades
gradient noise against hardware efficiency; a common heuristic scales $\eta$ with batch size. We use
the paper's pairing: $m=128$, $\eta=0.1$.

### 8.3 Momentum

Accumulate an exponentially decaying average of past gradients and step along that:

$$
v \leftarrow \mu\,v + \hat{g}, \qquad \theta \leftarrow \theta - \eta\,v, \qquad \mu = 0.9 .
$$

Intuition: a heavy ball rolling downhill. It damps oscillation across high-curvature directions and
builds speed along consistent low-curvature directions, which markedly accelerates convergence in
the ill-conditioned valleys typical of deep nets (Sutskever et al., 2013).

> **Exact PyTorch form** (`torch.optim.SGD`, dampening $0$): with weight decay $\lambda$ applied
> first, $g_t \leftarrow \hat g_t + \lambda\theta_{t-1}$; then $b_t \leftarrow \mu\,b_{t-1} + g_t$
> ($b_1 = g_1$); then $\theta_t \leftarrow \theta_{t-1} - \eta\,b_t$. Weight decay is therefore also
> carried in the momentum buffer. Nesterov (not used) evaluates the gradient at the look-ahead point
> $\theta - \eta\mu b$.

### 8.4 Weight decay

Add $\frac{\lambda}{2}\lVert\theta\rVert_2^2$ to the objective; its gradient is $\lambda\theta$, so
each step gains a term $-\eta\lambda\theta$ pulling weights toward zero. For plain SGD this is
exactly L2 regularization (for Adam it is not — hence AdamW; we use SGD, so the equivalence holds).
We use $\lambda = 10^{-4}$ (paper value).

**Interaction with BN.** Because BN makes the loss scale-invariant to the weights feeding it (§3),
weight decay on those layers does not act as classical capacity control — it shrinks $\lVert W\rVert$,
which *raises the effective learning rate* of that layer (the gradient's component that changes the
function scales as $1/\lVert W\rVert$). So in a BN network, weight decay and learning rate are
entangled (Van Laarhoven, 2017; Zhang et al., 2019). Worth keeping in mind when interpreting Exp 7
and Exp 8.

### 8.5 Learning-rate scheduling

A constant $\eta$ is a compromise: large enough for fast early progress, it is too large to settle
into a good minimum. In the noisy-quadratic model the loss floor is roughly proportional to
$\eta\times(\text{gradient-noise variance})$, so $\eta$ must shrink over time to converge.

| Schedule | Formula | Use here |
|----------|---------|----------|
| **Step** (default) | $\eta \leftarrow \gamma\,\eta$ at epochs $\{40, 60\}$, $\gamma = 0.1$ | core sweep; mirrors the paper's drops at 50% / 75% of training |
| **Cosine** | $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\big(1 + \cos(\pi t/T)\big)$ | Exp 8 arm; smooth decay, no milestone tuning (Loshchilov & Hutter, 2017) |
| **Constant** | $\eta_t = \eta_0$ | Exp 8 control |
| **Warmup** | linear ramp from small $\eta$ over the first steps, then base schedule | resnet-110 stretch runs only |

The staircase drops in a step-schedule loss curve (a sudden fall in loss right after each milestone)
are a useful sanity signal that the schedule is wired correctly.

### 8.6 Backpropagation (how the gradients above are obtained)

Reverse-mode automatic differentiation. One forward pass computes $\mathcal{L}$ and caches
activations. One backward pass walks the computation graph from $\mathcal{L}$ to the leaves; at each
node it multiplies the upstream $\partial\mathcal{L}/\partial(\text{output})$ by the node's local
Jacobian to get $\partial\mathcal{L}/\partial(\text{input})$ and
$\partial\mathcal{L}/\partial(\text{params})$, using the per-operation rules in §1–§7. All partials
are obtained in a single sweep at ~2× the forward cost. PyTorch's `autograd` builds the graph
dynamically during the forward pass; `loss.backward()` runs the sweep; `optimizer.step()` applies
§8.2–8.5.

---

## 9. How the math maps to project decisions

| Decision (spec) | Justification from this document |
|-----------------|----------------------------------|
| `bias=False` on all convs | BN's $\beta$ makes a conv bias redundant (§1, §3) |
| $3\times3$ convs, $P=1$ | "same" padding keeps stage resolution fixed (§1) |
| stride-2 conv for downsampling (no pooling) | matches the paper; output-size formula gives $32\to16\to8$ (§1) |
| He init, `fan_out` | preserves forward and backward variance through ReLU (§8.0) |
| ReLU after the residual add | keeps a clean-ish additive path for the gradient (§4, §5) |
| Option A shortcut as default | zero params ⇒ plain and residual are exactly parameter-matched (§1, §4) |
| `nn.CrossEntropyLoss` on logits | fused softmax+CE gradient $p - t$ is stable (§6, §7) |
| SGD + momentum 0.9 + wd 1e-4 | paper recipe; momentum for conditioning, wd interacts with BN (§8.3–8.4) |
| step LR ×0.1 at 40, 60 | noise-floor argument for decay; mirrors paper milestones (§8.5) |
| AMP off for Exp 5 / Exp 7 | gradient-norm and BN-ablation numbers must be clean fp32 (§3, §5) |

---

## References

- He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016. (the paper)
- He, Zhang, Ren, Sun. *Identity Mappings in Deep Residual Networks.* ECCV 2016. (the §5 derivation)
- He, Zhang, Ren, Sun. *Delving Deep into Rectifiers.* ICCV 2015. (He initialization)
- Ioffe, Szegedy. *Batch Normalization.* ICML 2015.
- Santurkar, Tsipras, Ilyas, Madry. *How Does Batch Normalization Help Optimization?* NeurIPS 2018.
- Li, Xu, Taylor, Studer, Goldstein. *Visualizing the Loss Landscape of Neural Nets.* NeurIPS 2018.
- Balduzzi et al. *The Shattered Gradients Problem.* ICML 2017.
- Sutskever, Martens, Dahl, Hinton. *On the importance of initialization and momentum.* ICML 2013.
- Loshchilov, Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.
- Loshchilov, Hutter. *Decoupled Weight Decay Regularization (AdamW).* ICLR 2019.
- van Laarhoven. *L2 Regularization versus Batch and Weight Normalization.* 2017.
- Goodfellow, Bengio, Courville. *Deep Learning.* MIT Press, 2016. (GD, backprop background)
