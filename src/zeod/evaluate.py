"""Validation metrics (mAP/precision/recall) plus lightweight FP/FN example mining."""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path

from zeod.config import AppConfig
from zeod.device import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class ValMetrics:
    map50: float
    map50_95: float
    precision: float
    recall: float


def evaluate_yolo(weights_path: Path, config: AppConfig) -> ValMetrics:
    from ultralytics import YOLO

    dataset_dir = config.resolve(config.paths.dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path} - run `zeod train` first")

    device = resolve_device(config.evaluate.device)
    model = YOLO(str(weights_path))
    # Deliberately NOT passing evaluate.conf here: mAP is computed by sweeping
    # the full confidence range internally, so a fixed display threshold like
    # 0.25 would truncate that sweep and silently zero out the metrics for an
    # early/undertrained model. evaluate.conf is used for analyze_errors and
    # `infer` below, where a fixed display threshold is exactly what's wanted.
    val_results = model.val(data=str(data_yaml), device=device, iou=config.evaluate.iou)

    metrics = ValMetrics(
        map50=float(val_results.box.map50),
        map50_95=float(val_results.box.map),
        precision=float(val_results.box.mp),
        recall=float(val_results.box.mr),
    )
    logger.info(
        "Validation: mAP50=%.4f mAP50-95=%.4f precision=%.4f recall=%.4f",
        metrics.map50,
        metrics.map50_95,
        metrics.precision,
        metrics.recall,
    )
    return metrics


def load_val_items(dataset_dir: Path) -> list[tuple[str, Path, list[str]]]:
    """Read back (name, image_path, gt_yolo_lines) for every image in dataset/val."""
    images_dir = dataset_dir / "val" / "images"
    labels_dir = dataset_dir / "val" / "labels"
    items = []
    for img_path_str in sorted(glob.glob(str(images_dir / "*"))):
        img_path = Path(img_path_str)
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
        items.append((img_path.stem, img_path, gt_lines))
    return items


def analyze_errors(
    model,
    val_items: list[tuple[str, Path, list[str]]],
    conf: float,
    device: str | int,
    n_examples: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Find example images where predicted box count over/under-shoots ground truth.

    This is a coarse count-based proxy for FP/FN (not IoU-matched), good
    enough to eyeball where the detector over- or under-predicts; it mirrors
    the original notebook's error-analysis approach.
    """
    fp_examples: list[dict] = []
    fn_examples: list[dict] = []

    for name, img_path, gt_lines in val_items:
        pred_results = model.predict(str(img_path), conf=conf, device=device, verbose=False)
        pred = pred_results[0]
        n_pred = len(pred.boxes) if pred.boxes is not None else 0
        n_gt = len(gt_lines)

        record = {"name": name, "image_path": str(img_path), "n_pred": n_pred, "n_gt": n_gt}
        if n_pred > n_gt and len(fp_examples) < n_examples:
            fp_examples.append(record)
        elif n_pred < n_gt and len(fn_examples) < n_examples:
            fn_examples.append(record)

        if len(fp_examples) >= n_examples and len(fn_examples) >= n_examples:
            break

    return fp_examples, fn_examples
