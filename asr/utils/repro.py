from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReproState:
    seed: int
    deterministic: bool


def set_reproducibility(seed: int, deterministic: bool) -> ReproState:
    """Best-effort reproducibility controls.

    Notes:
    - Some GPU kernels remain nondeterministic.
    - When deterministic=True, we enable deterministic algorithms in warn-only mode
      to avoid hard failures on unsupported ops.
    """

    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # cuBLAS determinism (only takes effect for certain GEMM paths)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older/newer PyTorch variants may not support warn_only
            torch.use_deterministic_algorithms(True)

    return ReproState(seed=seed, deterministic=deterministic)
