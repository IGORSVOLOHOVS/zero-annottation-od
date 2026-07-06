"""Assembles a YOLO-format dataset (train/val images + labels + data.yaml) from raw labels."""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from zeod.dataset.split import train_val_split
from zeod.dataset.yolo_convert import convert_to_yolo

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    n_total_labeled: int = 0
    n_valid: int = 0
    n_skipped: int = 0
    n_train: int = 0
    n_val: int = 0
    class_counts: Counter = field(default_factory=Counter)


DatasetItem = tuple[str, Path, list[str]]  # (name, image_path, yolo_lines)


def build_valid_items(
    labels_raw: dict,
    class_map: dict[str, int],
    min_box_frac: float,
    max_box_frac: float,
    dedup_iou_threshold: float | None = 0.9,
    head_crop_max_aspect_ratio: float | None = None,
) -> tuple[list[DatasetItem], DatasetStats]:
    """Turn raw VLM detections into (name, image_path, yolo_lines) items, dropping empties."""
    stats = DatasetStats(n_total_labeled=len(labels_raw))
    valid_items: list[DatasetItem] = []

    for name, data in labels_raw.items():
        detections = data.get("detections") or []
        if not detections:
            stats.n_skipped += 1
            continue

        img_path = Path(data["image_path"])
        if not img_path.exists():
            logger.warning("Image referenced in labels not found on disk: %s", img_path)
            stats.n_skipped += 1
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines = convert_to_yolo(
            detections,
            img_w,
            img_h,
            class_map,
            min_box_frac,
            max_box_frac,
            dedup_iou_threshold,
            head_crop_max_aspect_ratio,
        )
        if not yolo_lines:
            stats.n_skipped += 1
            continue

        valid_items.append((name, img_path, yolo_lines))
        for line in yolo_lines:
            stats.class_counts[int(line.split()[0])] += 1

    stats.n_valid = len(valid_items)
    return valid_items, stats


def _save_split(items: list[DatasetItem], dataset_dir: Path, split: str) -> None:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for name, img_path, yolo_lines in items:
        shutil.copy2(img_path, images_dir / f"{name}{img_path.suffix}")
        (labels_dir / f"{name}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")


def write_data_yaml(dataset_dir: Path, class_map: dict[str, int]) -> Path:
    names_by_id = {v: k for k, v in class_map.items()}
    lines = [f"path: {dataset_dir}", "train: train/images", "val: val/images", "names:"]
    for class_id in sorted(names_by_id):
        lines.append(f"  {class_id}: {names_by_id[class_id]}")
    data_yaml_path = dataset_dir / "data.yaml"
    data_yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml_path


def build_dataset(
    labels_raw: dict,
    dataset_dir: Path,
    class_map: dict[str, int],
    val_split: float,
    min_box_frac: float,
    max_box_frac: float,
    seed: int,
    dedup_iou_threshold: float | None = 0.9,
    head_crop_max_aspect_ratio: float | None = None,
) -> DatasetStats:
    """Convert raw labels into a full YOLO dataset directory, return summary stats."""
    valid_items, stats = build_valid_items(
        labels_raw, class_map, min_box_frac, max_box_frac, dedup_iou_threshold, head_crop_max_aspect_ratio
    )
    if not valid_items:
        logger.error(
            "No valid annotations found (out of %d labeled images) - nothing to build", stats.n_total_labeled
        )
        return stats

    train_items, val_items = train_val_split(valid_items, val_split, seed)
    stats.n_train, stats.n_val = len(train_items), len(val_items)

    _save_split(train_items, dataset_dir, "train")
    _save_split(val_items, dataset_dir, "val")
    write_data_yaml(dataset_dir, class_map)

    logger.info(
        "Dataset built at %s: %d train / %d val (skipped %d/%d unlabeled or invalid)",
        dataset_dir,
        stats.n_train,
        stats.n_val,
        stats.n_skipped,
        stats.n_total_labeled,
    )
    return stats
