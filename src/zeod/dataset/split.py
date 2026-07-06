"""Deterministic train/val split, independent of sklearn so tests need no extra deps."""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


def train_val_split(items: list[T], val_split: float, seed: int) -> tuple[list[T], list[T]]:
    """Shuffle ``items`` with a seeded RNG and split off ``val_split`` fraction.

    Same (items, val_split, seed) always yields the same split - the RNG is
    local to this call and never mutates global `random` state, so calling
    it repeatedly (e.g. in tests) is safe and reproducible.
    """
    if not 0 < val_split < 1:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    n_val = round(len(shuffled) * val_split)
    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]
    return train_items, val_items
