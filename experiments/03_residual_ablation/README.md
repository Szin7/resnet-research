# Shortcut removal

The existing Plain-56 and ResNet-56 runs already provide a direct comparison of the block with and without additive shortcuts. Both have 853,018 learnable parameters and use the same training recipe.

| Model | Final training error | Test accuracy |
|---|---:|---:|
| Plain-56 (no shortcut) | 19.518% | 79.43% |
| ResNet-56 (shortcut A) | 1.036% | 91.68% |

This shows a large difference in the completed seed-1 runs. It does not by itself prove that every optimization difference comes from one specific mechanism.

The earlier plan also included a separate confirmation run with shortcuts explicitly zeroed at runtime. **That extra confirmation was not run.** The results above reuse the already completed depth experiment.
