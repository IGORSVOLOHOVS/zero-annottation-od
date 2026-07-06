"""Fine-tunes a YOLOv8 detector on the auto-labeled dataset."""

from __future__ import annotations

import logging
from pathlib import Path

from zeod.config import AppConfig
from zeod.device import resolve_device

logger = logging.getLogger(__name__)


def train_yolo(config: AppConfig):
    from ultralytics import YOLO

    dataset_dir = config.resolve(config.paths.dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found - run `zeod build-dataset` first")

    runs_dir = config.resolve(config.paths.runs_dir)
    device = resolve_device(config.train.device)
    logger.info("Training YOLO on device=%s for %d epochs", device, config.train.epochs)

    model = YOLO(config.train.base_weights)
    results = model.train(
        data=str(data_yaml),
        epochs=config.train.epochs,
        imgsz=config.train.imgsz,
        batch=config.train.batch,
        patience=config.train.patience,
        device=device,
        project=str(runs_dir),
        name=config.train.experiment_name,
        exist_ok=True,
        seed=config.seed,
    )
    logger.info("Training finished, weights at %s", best_weights_path(config))
    return results


def best_weights_path(config: AppConfig) -> Path:
    runs_dir = config.resolve(config.paths.runs_dir)
    return runs_dir / config.train.experiment_name / "weights" / "best.pt"
