"""Runs the trained YOLO detector on a directory or glob of images and saves annotated copies."""

from __future__ import annotations

import logging
from pathlib import Path

from zeod.device import resolve_device

logger = logging.getLogger(__name__)


def run_inference(
    weights_path: Path,
    image_paths: list[Path],
    output_dir: Path,
    conf: float = 0.25,
    device: str = "auto",
) -> list[Path]:
    from ultralytics import YOLO

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path} - run `zeod train` first")
    if not image_paths:
        raise ValueError("No images to run inference on")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    model = YOLO(str(weights_path))

    saved_paths = []
    for img_path in image_paths:
        results = model.predict(str(img_path), conf=conf, device=resolved_device, verbose=False)
        out_path = output_dir / img_path.name
        results[0].save(filename=str(out_path))
        saved_paths.append(out_path)

    logger.info("Saved %d annotated images to %s", len(saved_paths), output_dir)
    return saved_paths
