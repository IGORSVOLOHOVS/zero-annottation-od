"""Point 7: benchmarks, so a performance regression shows up as a number.

The labelling run is the long pole in this project: a VLM looks at every raw
photo and its answer is then parsed, deduplicated and converted to YOLO format.
The model call dominates wall-clock time and is not measured here — it depends
on the machine's GPU and on which weights are loaded, so a figure including it
says nothing about this code.

What is measured is the per-image work around the model, and one function in
particular. `_dedup_by_iou` compares each detection against every detection
already kept, so its cost grows with the square of the count. On a photo with
three people that is invisible; on a crowded construction site it is not. The
benchmark makes the shape visible rather than leaving it to be discovered.

    pytest benchmarks --benchmark-only
    pytest benchmarks --benchmark-only --benchmark-compare
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from zeod.dataset.split import train_val_split
from zeod.dataset.yolo_convert import _dedup_by_iou
from zeod.labeling.grid import FACTOR, smart_resize
from zeod.labeling.parser import parse_json_response


def detections(count: int, *, overlapping: bool) -> list[tuple[str, list[float]]]:
    """Boxes that either all overlap heavily or none do.

    The two cases take different paths through the deduplication loop: with
    distinct boxes every comparison runs to the end, which is the worst case.
    """
    if overlapping:
        return [("helmet", [100.0, 100.0, 200.0, 200.0]) for _ in range(count)]
    return [("helmet", [i * 50.0, 0.0, i * 50.0 + 40.0, 40.0]) for i in range(count)]


@pytest.mark.parametrize("count", [4, 32, 128])
def test_dedup_worst_case_is_quadratic(benchmark: Any, count: int) -> None:
    """Distinct boxes: every candidate is compared against everything kept."""
    items = detections(count, overlapping=False)

    kept = benchmark(_dedup_by_iou, items, 0.5)

    assert len(kept) == count


def test_dedup_when_everything_collapses(benchmark: Any) -> None:
    """Identical boxes: the loop exits early, so this is the cheap case."""
    items = detections(128, overlapping=True)

    kept = benchmark(_dedup_by_iou, items, 0.5)

    assert len(kept) == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_parsing_a_model_response(benchmark: Any, wrapped: bool) -> None:
    """Parsing runs once per image; markdown fences are the common VLM quirk."""
    payload = json.dumps([{"label": "helmet", "bbox_2d": [10, 20, 110, 120]} for _ in range(16)])
    text = f"Here are the detections:\n```json\n{payload}\n```" if wrapped else payload

    parsed = benchmark(parse_json_response, text)

    assert len(parsed) == 16


def test_parsing_rejects_junk_cheaply(benchmark: Any) -> None:
    """A bad response must not stop the run, and must not cost much either."""
    parsed = benchmark(parse_json_response, "I could not find any helmets in this image.")

    assert parsed == []


def test_smart_resize_is_arithmetic(benchmark: Any) -> None:
    """Called per image to fit the VLM's patch grid; should be free."""
    grid_w, grid_h = benchmark(smart_resize, 1920, 1080, 1024 * FACTOR**2, 16384 * FACTOR**2)

    assert grid_w > 0 and grid_h > 0


def test_split_scales_with_dataset(benchmark: Any) -> None:
    """Run once per dataset build, over every labelled image."""
    items = list(range(10_000))

    train, val = benchmark(train_val_split, items, 0.2, 42)

    assert len(train) + len(val) == len(items)
