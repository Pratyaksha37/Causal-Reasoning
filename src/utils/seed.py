"""Reproducibility utilities for setting and recording random seeds."""

import logging
import os
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> dict[str, Any]:
    """
    Set random seeds across Python's random, NumPy, and PyTorch (if available).

    Args:
        seed: Integer seed to set. Default is 42.

    Returns:
        Dict documenting the seed applied to each framework and determinism status.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    info: dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "pytorch": False,
        "pytorch_cuda": False,
        "deterministic_flag_set": False,
    }

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            info["pytorch_cuda"] = True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon MPS
            info["pytorch_mps"] = True

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        info["pytorch"] = True
        info["deterministic_flag_set"] = True
    except ImportError:
        pass

    logger.info("Random seed set to %d. Status: %s", seed, info)
    return info
