"""Optional interpretability (Phase 18, compute permitting).

Feature-map and Grad-CAM visualizations. Any write-up must use careful wording -- these show where
activations are strong, not that the model "understands" anything (see the master plan, section 21).

Implemented in Phase 18 if time allows; not on the critical path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def extract_feature_maps(
    model: nn.Module,
    image: torch.Tensor,
    *,
    layers: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Forward-hook the named layers and return their activation tensors for a single image."""
    raise NotImplementedError("extract_feature_maps: Phase 18")


def grad_cam(
    model: nn.Module,
    image: torch.Tensor,
    *,
    target_layer: str,
    target_class: int | None = None,
) -> np.ndarray:
    """Grad-CAM heatmap (Selvaraju et al., 2017) for one image and one target layer."""
    raise NotImplementedError("grad_cam: Phase 18")
