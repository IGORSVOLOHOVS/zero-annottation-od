"""Conversion of raw VLM detections (pixel-space bbox) to YOLO label lines."""

from __future__ import annotations


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedup_by_iou(items: list[tuple[str, list[float]]], iou_threshold: float) -> list[tuple[str, list[float]]]:
    """Drop later (label, bbox) pairs that heavily overlap (same label, IoU > threshold) an earlier one.

    A VLM occasionally emits near-identical repeat detections for the same
    person; without this, YOLO sees two "ground truth" boxes for one head.
    """
    kept: list[tuple[str, list[float]]] = []
    for label, bbox in items:
        if any(label == kl and _iou(bbox, kb) > iou_threshold for kl, kb in kept):
            continue
        kept.append((label, bbox))
    return kept


def _crop_tall_box_to_head(bbox: list[float], max_aspect_ratio: float) -> list[float]:
    """If a box is much taller than wide (height/width > max_aspect_ratio), crop it to a
    square anchored at the top edge.

    Empirically, this VLM draws consistently tight, roughly-square head boxes
    for "helmet" detections (median height/width ~0.9) but frequently falls
    back to full-body boxes for "no_helmet" (median ~2.5) despite the prompt
    explicitly asking for head-only boxes in both cases. Since the head is
    the topmost part of a person in every pose seen in this dataset (standing,
    crouching, bending), squaring the box using its own width and anchoring
    at the top is a reasonable proxy for "just the head" - not exact (the
    detected box's width is sometimes shoulder-width, not head-width, so the
    result can still be head+shoulders), but it is a clear improvement over
    a box spanning the entire body. See README for the measured effect.
    """
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    if w <= 0 or h <= 0:
        return bbox
    if h / w > max_aspect_ratio:
        y_max = y_min + w
    return [x_min, y_min, x_max, y_max]


def convert_to_yolo(
    detections: list[dict],
    img_w: int,
    img_h: int,
    class_map: dict[str, int],
    min_box_frac: float = 0.005,
    max_box_frac: float = 0.95,
    dedup_iou_threshold: float | None = 0.9,
    head_crop_max_aspect_ratio: float | None = None,
) -> list[str]:
    """Convert [x_min, y_min, x_max, y_max] pixel boxes to normalized YOLO lines.

    Boxes are clamped to the image bounds before normalization. Detections
    with an unknown label, malformed bbox, degenerate (zero/negative) area
    after clamping, or a normalized width/height outside
    (min_box_frac, max_box_frac) are silently dropped - a VLM occasionally
    hallucinates a box covering almost the whole image or a sliver a few
    pixels wide, and those are not useful training signal.

    ``dedup_iou_threshold`` drops near-identical repeat detections (same
    label, high IoU); set to ``None`` to disable. ``head_crop_max_aspect_ratio``
    optionally crops overly-tall boxes toward a head-sized square (see
    ``_crop_tall_box_to_head``); ``None`` disables it.
    """
    parsed: list[tuple[str, list[float]]] = []
    for det in detections:
        label = str(det.get("label", "")).lower().strip()
        bbox = det.get("bbox", [])

        if label not in class_map or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            parsed.append((label, [float(v) for v in bbox]))
        except (TypeError, ValueError):
            continue

    if dedup_iou_threshold is not None:
        parsed = _dedup_by_iou(parsed, dedup_iou_threshold)

    yolo_lines = []
    for label, (x_min, y_min, x_max, y_max) in parsed:
        x_min = max(0.0, min(x_min, img_w))
        y_min = max(0.0, min(y_min, img_h))
        x_max = max(0.0, min(x_max, img_w))
        y_max = max(0.0, min(y_max, img_h))

        if x_max <= x_min or y_max <= y_min:
            continue

        if head_crop_max_aspect_ratio is not None:
            x_min, y_min, x_max, y_max = _crop_tall_box_to_head(
                [x_min, y_min, x_max, y_max], head_crop_max_aspect_ratio
            )

        cx = ((x_min + x_max) / 2) / img_w
        cy = ((y_min + y_max) / 2) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        if not (min_box_frac < w < max_box_frac and min_box_frac < h < max_box_frac):
            continue

        yolo_lines.append(f"{class_map[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines
