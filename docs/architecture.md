# Architecture

_Status: outline. Filled in Phase 8, once the model is implemented and its shapes verified._

Extends the frozen spec ([project_specification.md](project_specification.md) §4) with explanation,
diagrams, and design rationale.

Planned contents:

- Stem → three stages → head, with a per-layer table (channels, spatial size, stride, parameters).
- The BasicBlock: conv/BN/ReLU ordering and why ReLU sits after the residual add.
- Shortcut options A/B/C with worked tensor shapes at the stage-2 and stage-3 transitions.
- The `residual=False` (plain) path and why it is parameter-identical to the residual path.
- Where He initialization is applied and the `zero_init_residual` option.
- A diagram of resnet-20 vs plain-20 side by side.
- Cross-check of computed vs `torchinfo`-reported parameter counts for every depth.
