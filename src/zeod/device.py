"""Shared CUDA/CPU device resolution for training, evaluation, and inference."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str | int:
    """Resolve "auto" to a concrete Ultralytics device spec.

    Falls back to CPU (slower, but functional) whenever CUDA isn't available,
    so the pipeline runs end-to-end on machines without an NVIDIA GPU.
    """
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return 0
    logger.warning("CUDA not available - falling back to CPU (this will be significantly slower)")
    return "cpu"
