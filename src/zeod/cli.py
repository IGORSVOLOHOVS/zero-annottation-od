"""Command-line entry point: `zeod <command> [--config path] [--set key=value ...]`."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

from zeod.config import AppConfig, load_config
from zeod.logging_setup import setup_logging

logger = logging.getLogger(__name__)

_WEIGHTS_HELP = "Path to a .pt file (default: runs/<name>/weights/best.pt)"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _image_paths(config: AppConfig) -> list[Path]:
    images_dir = config.resolve(config.paths.images_dir)
    paths = sorted(Path(p) for p in glob.glob(str(images_dir / config.paths.image_glob)))
    if not paths:
        raise FileNotFoundError(f"No images matching {config.paths.image_glob!r} in {images_dir}")
    return paths


def cmd_label(config: AppConfig, args: argparse.Namespace) -> None:
    from zeod.labeling.backend import LlamaCppServerBackend
    from zeod.labeling.pipeline import label_images

    image_paths = _image_paths(config)
    if args.limit:
        image_paths = image_paths[: args.limit]
    logger.info("Found %d images to label", len(image_paths))

    labels_raw_path = config.resolve(config.paths.labels_raw_path)
    with LlamaCppServerBackend(config.llama_cpp) as backend:
        backend.start()
        label_images(
            image_paths=image_paths,
            backend=backend,
            system_prompt=config.labeling.system_prompt,
            user_prompt=config.labeling.user_prompt,
            temperature=config.labeling.temperature,
            max_tokens=config.labeling.max_tokens,
            labels_raw_path=labels_raw_path,
            checkpoint_every=config.labeling.checkpoint_every,
            bbox_rescaler=backend.bbox_rescaler(),
        )


def cmd_build_dataset(config: AppConfig, args: argparse.Namespace) -> None:
    from zeod.dataset.build import build_dataset
    from zeod.labeling.pipeline import load_labels_raw

    labels_raw_path = config.resolve(config.paths.labels_raw_path)
    labels_raw = load_labels_raw(labels_raw_path)
    if not labels_raw:
        raise FileNotFoundError(f"No labels found at {labels_raw_path} - run `zeod label` first")

    stats = build_dataset(
        labels_raw=labels_raw,
        dataset_dir=config.resolve(config.paths.dataset_dir),
        class_map=config.labeling.class_map,
        val_split=config.dataset.val_split,
        min_box_frac=config.dataset.min_box_frac,
        max_box_frac=config.dataset.max_box_frac,
        seed=config.seed,
        dedup_iou_threshold=config.dataset.dedup_iou_threshold,
        head_crop_max_aspect_ratio=config.dataset.head_crop_max_aspect_ratio,
    )
    print(json.dumps(stats.__dict__, default=dict, indent=2))


def cmd_train(config: AppConfig, args: argparse.Namespace) -> None:
    from zeod.train import train_yolo

    train_yolo(config)


def cmd_evaluate(config: AppConfig, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    from zeod.evaluate import analyze_errors, evaluate_yolo, load_val_items
    from zeod.train import best_weights_path

    weights_path = Path(args.weights) if args.weights else best_weights_path(config)
    metrics = evaluate_yolo(weights_path, config)
    print(json.dumps(metrics.__dict__, indent=2))

    if args.analyze_errors:
        from zeod.device import resolve_device

        model = YOLO(str(weights_path))
        val_items = load_val_items(config.resolve(config.paths.dataset_dir))
        fp_examples, fn_examples = analyze_errors(
            model,
            val_items,
            conf=config.evaluate.conf,
            device=resolve_device(config.evaluate.device),
            n_examples=config.evaluate.fp_fn_examples,
        )
        errors = {"false_positive_examples": fp_examples, "false_negative_examples": fn_examples}
        print(json.dumps(errors, indent=2))


def cmd_infer(config: AppConfig, args: argparse.Namespace) -> None:
    from zeod.infer import run_inference
    from zeod.train import best_weights_path

    weights_path = Path(args.weights) if args.weights else best_weights_path(config)
    image_paths = [Path(p) for p in glob.glob(args.images)]
    run_inference(
        weights_path=weights_path,
        image_paths=image_paths,
        output_dir=Path(args.output),
        conf=args.conf,
        device=config.evaluate.device,
    )


def cmd_pipeline(config: AppConfig, args: argparse.Namespace) -> None:
    cmd_label(config, args)
    cmd_build_dataset(config, args)
    cmd_train(config, args)
    args.analyze_errors = True
    args.weights = None
    cmd_evaluate(config, args)


def build_parser() -> argparse.ArgumentParser:
    # --config/--set live ONLY on the top-level parser and must come before the
    # subcommand (`zeod --set train.epochs=5 train`, not `zeod train --set ...`).
    # argparse's subparser dispatch unconditionally overwrites shared dests with
    # its own (default) values even when parented via `parents=`, so duplicating
    # these options onto every subparser would silently discard the top-level
    # value instead of merging it - keeping them in one place avoids that trap.
    parser = argparse.ArgumentParser(prog="zeod", description="Zero-Annotation-OD pipeline CLI")
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: configs/default.yaml)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config value, e.g. --set train.epochs=5 (repeatable)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_label = sub.add_parser("label", help="Auto-label images with the local VLM backend")
    p_label.add_argument("--limit", type=int, default=None, help="Only label the first N images (smoke tests)")
    p_label.set_defaults(func=cmd_label)

    p_build = sub.add_parser("build-dataset", help="Convert labels_raw.json into a YOLO dataset")
    p_build.set_defaults(func=cmd_build_dataset)

    p_train = sub.add_parser("train", help="Fine-tune YOLOv8 on the built dataset")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Compute val metrics and optionally mine FP/FN examples")
    p_eval.add_argument("--weights", default=None, help=_WEIGHTS_HELP)
    p_eval.add_argument("--analyze-errors", action="store_true", help="Also print FP/FN example images")
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = sub.add_parser("infer", help="Run the trained detector on new images")
    p_infer.add_argument("images", help="Glob pattern for input images, e.g. 'images/*.png'")
    p_infer.add_argument("--weights", default=None, help=_WEIGHTS_HELP)
    p_infer.add_argument("--output", default="runs/infer_output", help="Directory for annotated output images")
    p_infer.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_infer.set_defaults(func=cmd_infer)

    p_pipeline = sub.add_parser("pipeline", help="Run label -> build-dataset -> train -> evaluate end to end")
    p_pipeline.add_argument("--limit", type=int, default=None, help="Only label the first N images (smoke tests)")
    p_pipeline.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, args.overrides)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    _seed_everything(config.seed)

    try:
        args.func(config, args)
    except (FileNotFoundError, ValueError, TimeoutError, RuntimeError) as e:
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
