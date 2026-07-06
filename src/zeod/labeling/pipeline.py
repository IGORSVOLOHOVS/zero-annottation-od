"""Drives the auto-labeling step: image -> VLM backend -> parsed detections -> JSON."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from zeod.labeling.backend import VLMBackend
from zeod.labeling.parser import parse_json_response

logger = logging.getLogger(__name__)


def _apply_bbox_rescaler(
    detections: list[dict], image_path: Path, rescaler: Callable[[list[float], int, int], list[float]]
) -> list[dict]:
    """Rescale each detection's bbox in place using ``rescaler(bbox, orig_w, orig_h)``.

    Detections with a missing/malformed bbox are passed through unchanged;
    convert_to_yolo already drops those defensively at dataset-build time.
    """
    try:
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size
    except OSError:
        return detections

    rescaled = []
    for det in detections:
        bbox = det.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                det = {**det, "bbox": rescaler([float(v) for v in bbox], orig_w, orig_h)}
            except (TypeError, ValueError):
                pass
        rescaled.append(det)
    return rescaled


def load_labels_raw(labels_raw_path: Path) -> dict:
    if labels_raw_path.exists():
        with open(labels_raw_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels_raw(labels_raw: dict, labels_raw_path: Path) -> None:
    labels_raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_raw_path, "w", encoding="utf-8") as f:
        json.dump(labels_raw, f, indent=2, ensure_ascii=False)


def _label_one(
    backend: VLMBackend,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    bbox_rescaler: Callable[[list[float], int, int], list[float]] | None,
) -> tuple[str, list[dict], str | None]:
    """Try labeling one image; returns (raw_response, detections, error_or_None)."""
    try:
        response_text = backend.generate(image_path, system_prompt, user_prompt, temperature, max_tokens)
        detections = parse_json_response(response_text)
        if bbox_rescaler is not None and detections:
            detections = _apply_bbox_rescaler(detections, image_path, bbox_rescaler)
        return response_text, detections, None
    except Exception as e:  # noqa: BLE001 - one bad image must not kill the run
        return "", [], str(e)


def label_images(
    image_paths: list[Path],
    backend: VLMBackend,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    labels_raw_path: Path,
    checkpoint_every: int = 20,
    resume: bool = True,
    bbox_rescaler: Callable[[list[float], int, int], list[float]] | None = None,
    max_consecutive_failures: int = 3,
) -> dict:
    """Label every image not already present in ``labels_raw_path``.

    Results are checkpointed to disk every ``checkpoint_every`` images so a
    crash or interrupted run doesn't lose already-computed labels; re-running
    with ``resume=True`` (default) skips images already in the file.

    If ``max_consecutive_failures`` requests in a row fail (e.g. a llama-server
    process that's wedged - observed in practice: /health kept responding but
    /v1/chat/completions requests hung indefinitely after ~100 requests), the
    backend is restarted (``backend.restart()``) and the current image is
    retried once before moving on, so a long run can self-heal instead of
    silently producing hundreds of empty labels.
    """
    labels_raw = load_labels_raw(labels_raw_path) if resume else {}
    # A prior backend failure (timeout, connection refused, ...) stores an
    # "error" field so it's retried on resume; a genuine "no detections"
    # response has no "error" key and is treated as done.
    pending = [p for p in image_paths if p.stem not in labels_raw or labels_raw[p.stem].get("error")]

    if not pending:
        logger.info("All %d images already labeled in %s", len(image_paths), labels_raw_path)
        return labels_raw

    logger.info(
        "Labeling %d/%d images (skipping %d already done)", len(pending), len(image_paths), len(labels_raw)
    )

    n_errors = 0
    consecutive_failures = 0
    for i, image_path in enumerate(tqdm(pending, desc="Labeling")):
        response_text, detections, error = _label_one(
            backend, image_path, system_prompt, user_prompt, temperature, max_tokens, bbox_rescaler
        )

        if error is not None:
            consecutive_failures += 1
            logger.warning("Failed to label %s: %s", image_path.name, error)
            if consecutive_failures >= max_consecutive_failures:
                try:
                    backend.restart()
                    consecutive_failures = 0
                    response_text, detections, error = _label_one(
                        backend, image_path, system_prompt, user_prompt, temperature, max_tokens, bbox_rescaler
                    )
                except Exception as e:  # noqa: BLE001 - restart failing shouldn't kill the whole run
                    logger.warning("Backend restart failed: %s", e)
        else:
            consecutive_failures = 0

        if error is not None:
            n_errors += 1

        entry = {
            "image_path": str(image_path),
            "raw_response": response_text,
            "detections": detections,
        }
        if error is not None:
            entry["error"] = error
        labels_raw[image_path.stem] = entry

        if (i + 1) % checkpoint_every == 0:
            save_labels_raw(labels_raw, labels_raw_path)

    save_labels_raw(labels_raw, labels_raw_path)

    if n_errors:
        logger.warning(
            "%d/%d images failed during labeling and were stored with empty detections", n_errors, len(pending)
        )

    n_with_det = sum(1 for v in labels_raw.values() if v["detections"])
    logger.info("Labeling done: %d/%d images have at least one detection", n_with_det, len(labels_raw))
    return labels_raw
