"""Typed configuration for the pipeline, loaded from YAML with optional overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class PathsConfig(BaseModel):
    images_dir: Path = Path("images")
    image_glob: str = "hard_hat_workers*.png"
    labels_raw_path: Path = Path("labels_raw.json")
    dataset_dir: Path = Path("dataset")
    runs_dir: Path = Path("runs")


class LabelingConfig(BaseModel):
    class_map: dict[str, int] = Field(default_factory=lambda: {"helmet": 0, "no_helmet": 1})
    system_prompt: str
    user_prompt: str
    temperature: float = 0.1
    max_tokens: int = 512
    checkpoint_every: int = 20


class LlamaCppConfig(BaseModel):
    server_binary: str = "llama-server"
    host: str = "127.0.0.1"
    port: int = 8090
    hf_repo: str | None = "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF"
    hf_quant: str = "Q4_K_M"
    model_path: Path | None = None
    mmproj_path: Path | None = None
    n_gpu_layers: int = -1
    ctx_size: int = 4096
    # zeod's labeling loop is strictly sequential (one request at a time), so
    # multiple server slots buy nothing and were observed to make the server
    # hang after ~100 requests during a real 992-image run (health check kept
    # responding, but /v1/chat/completions requests started timing out
    # indefinitely - likely a slot-scheduling issue). Pinning to 1 slot matches
    # actual usage and avoided any recurrence in testing.
    n_parallel: int = 1
    # llama.cpp's own startup warning for Qwen-VL models: grounding/bbox accuracy
    # degrades badly below 1024 image tokens. See --image-min-tokens in `llama-server --help`.
    image_min_tokens: int = 1024
    # None = use Qwen's own default (16384 tokens). Needed (together with
    # image_min_tokens) to reconstruct the vision-grid size for bbox rescaling;
    # see zeod.labeling.grid.
    image_max_tokens: int | None = None
    startup_timeout_s: int = 240
    request_timeout_s: int = 120
    extra_server_args: list[str] = Field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def min_pixels(self) -> int:
        from zeod.labeling.grid import FACTOR

        return self.image_min_tokens * FACTOR * FACTOR

    @property
    def max_pixels(self) -> int:
        from zeod.labeling.grid import DEFAULT_MAX_TOKENS, FACTOR

        tokens = self.image_max_tokens if self.image_max_tokens is not None else DEFAULT_MAX_TOKENS
        return tokens * FACTOR * FACTOR


class DatasetConfig(BaseModel):
    val_split: float = 0.2
    min_box_frac: float = 0.005
    max_box_frac: float = 0.95
    # Drop near-identical repeat detections (same label, IoU above this) - set
    # to null to disable. See zeod.dataset.yolo_convert._dedup_by_iou.
    dedup_iou_threshold: float | None = 0.9
    # Qwen2.5-VL draws tight, roughly-square head boxes for "helmet" (median
    # height/width ~0.9 in our data) but frequently falls back to full-body
    # boxes for "no_helmet" (median ~2.5) despite the prompt asking for
    # head-only boxes either way. Boxes taller than `width * this ratio` get
    # cropped to a top-anchored square as an approximate head-only fix. Set to
    # null to disable. See zeod.dataset.yolo_convert._crop_tall_box_to_head
    # and README for the measured before/after effect.
    head_crop_max_aspect_ratio: float | None = 1.8


class TrainConfig(BaseModel):
    base_weights: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    patience: int = 10
    device: str = "auto"
    experiment_name: str = "helmet_detection"


class EvaluateConfig(BaseModel):
    conf: float = 0.25
    iou: float = 0.5
    device: str = "auto"
    fp_fn_examples: int = 4


class AppConfig(BaseModel):
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    labeling: LabelingConfig
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)

    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve a config-relative path against the project root."""
        return path if path.is_absolute() else self.project_root / path


def _apply_dotted_override(data: dict[str, Any], dotted_key: str, value: str) -> None:
    keys = dotted_key.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    # best-effort type coercion so `--set train.epochs=5` works without quotes
    parsed: Any = value
    for caster in (int, float):
        try:
            parsed = caster(value)
            break
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        parsed = value.lower() == "true"
    if value.lower() in ("null", "none"):
        parsed = None
    node[keys[-1]] = parsed


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    """Load YAML config, apply dotted-key overrides (e.g. "train.epochs=5"), validate."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid --set override {override!r}, expected key=value")
        key, value = override.split("=", 1)
        _apply_dotted_override(data, key.strip(), value.strip())

    config = AppConfig(**data)
    if project_root is not None:
        config.project_root = project_root
    return config
