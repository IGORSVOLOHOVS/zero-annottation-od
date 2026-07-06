"""Rescales Qwen2-VL/2.5-VL bbox output from the model's internal vision grid to real pixels.

Qwen2-VL-family models use "dynamic resolution": before the vision encoder
sees an image, it's resized so its token count falls within
[min_tokens, max_tokens], with both dimensions rounded to a multiple of
``factor`` (patch_size * spatial_merge_size = 14 * 2 = 28 for Qwen2/2.5-VL).
The model was trained to emit bbox coordinates in *that resized grid*, not
in the original image's pixel space - and it does this rigidly regardless
of prompt instructions asking for normalized floats or "pixel coordinates"
(verified empirically: asking for [0,1] fractions still returned grid-space
integers). This module reimplements Qwen's public ``smart_resize`` algorithm
(the same one the original vLLM-based pipeline used via ``qwen_vl_utils``)
so we can map coordinates back to true image pixels.

This is a best-effort reconstruction of llama.cpp's internal C++ resize,
not a value read back from the server (the API doesn't expose it). It
matched empirical test coordinates closely in manual testing, but if boxes
still look systematically off after labeling, spot-check with the EDA
notebook's ``visualize_annotations`` cell before trusting a full run.
"""

from __future__ import annotations

import math

FACTOR = 28  # patch_size(14) * spatial_merge_size(2), Qwen2-VL / Qwen2.5-VL
DEFAULT_MAX_TOKENS = 16384  # Qwen's own IMAGE_MAX_TOKEN_NUM default


def _round_by_factor(x: float, factor: int) -> int:
    return max(factor, round(x / factor) * factor)


def _floor_by_factor(x: float, factor: int) -> int:
    return max(factor, math.floor(x / factor) * factor)


def _ceil_by_factor(x: float, factor: int) -> int:
    return math.ceil(x / factor) * factor


def smart_resize(
    width: int, height: int, min_pixels: int, max_pixels: int, factor: int = FACTOR
) -> tuple[int, int]:
    """Return (grid_width, grid_height): the resized dimensions Qwen-VL's vision encoder targets."""
    w_bar = _round_by_factor(width, factor)
    h_bar = _round_by_factor(height, factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        w_bar = _floor_by_factor(width / beta, factor)
        h_bar = _floor_by_factor(height / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        w_bar = _ceil_by_factor(width * beta, factor)
        h_bar = _ceil_by_factor(height * beta, factor)

    return w_bar, h_bar


def rescale_bbox_to_pixels(
    bbox: list[float],
    orig_w: int,
    orig_h: int,
    grid_w: int,
    grid_h: int,
) -> list[float]:
    """Map a [x_min, y_min, x_max, y_max] box from Qwen's vision grid to original pixel space."""
    scale_x, scale_y = orig_w / grid_w, orig_h / grid_h
    x_min, y_min, x_max, y_max = bbox
    return [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]
