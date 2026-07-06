"""End-to-end smoke test: label (mock VLM) -> build-dataset, on a handful of synthetic images.

No GPU, no network, no real model - validates that the pieces plug into each
other correctly, not model quality.
"""

import json

from PIL import Image

from zeod.dataset.build import build_dataset
from zeod.labeling.backend import MockBackend, VLMBackend
from zeod.labeling.pipeline import label_images, load_labels_raw

SYSTEM_PROMPT = "system prompt"
USER_PROMPT = "user prompt"
CLASS_MAP = {"helmet": 0, "no_helmet": 1}


def _make_images(tmp_path, n=7):
    paths = []
    for i in range(n):
        path = tmp_path / f"hard_hat_workers{i}.png"
        Image.new("RGB", (320, 240), color=(100, 100, 100)).save(path)
        paths.append(path)
    return paths


def _fake_vlm_response(image_path):
    """Mimics a real VLM: valid JSON for most images, one empty, one malformed."""
    idx = int("".join(filter(str.isdigit, image_path.stem)))
    if idx == 0:
        return "[]"  # no people detected
    if idx == 1:
        return "not even json"  # malformed response
    return json.dumps([{"label": "helmet" if idx % 2 else "no_helmet", "bbox": [20, 20, 80, 80]}])


def test_full_pipeline_smoke(tmp_path):
    image_paths = _make_images(tmp_path, n=7)
    labels_raw_path = tmp_path / "labels_raw.json"

    backend = MockBackend(responder=_fake_vlm_response)
    labels_raw = label_images(
        image_paths=image_paths,
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        temperature=0.1,
        max_tokens=512,
        labels_raw_path=labels_raw_path,
        checkpoint_every=3,
    )

    assert labels_raw_path.exists()
    assert len(labels_raw) == 7
    # image 0 -> empty, image 1 -> malformed -> both parsed to no detections
    assert labels_raw["hard_hat_workers0"]["detections"] == []
    assert labels_raw["hard_hat_workers1"]["detections"] == []
    # the rest have one detection each
    assert all(labels_raw[f"hard_hat_workers{i}"]["detections"] for i in range(2, 7))

    dataset_dir = tmp_path / "dataset"
    stats = build_dataset(
        labels_raw=labels_raw,
        dataset_dir=dataset_dir,
        class_map=CLASS_MAP,
        val_split=0.2,
        min_box_frac=0.005,
        max_box_frac=0.95,
        seed=42,
    )

    assert stats.n_valid == 5  # 7 total - 2 with no usable detections
    assert stats.n_train + stats.n_val == 5
    assert (dataset_dir / "data.yaml").exists()

    total_images = len(list((dataset_dir / "train" / "images").glob("*"))) + len(
        list((dataset_dir / "val" / "images").glob("*"))
    )
    assert total_images == 5


def test_label_images_is_resumable(tmp_path):
    image_paths = _make_images(tmp_path, n=4)
    labels_raw_path = tmp_path / "labels_raw.json"

    calls = []

    def counting_responder(image_path):
        calls.append(image_path)
        return "[]"

    backend = MockBackend(responder=counting_responder)
    label_images(image_paths, backend, SYSTEM_PROMPT, USER_PROMPT, 0.1, 512, labels_raw_path, checkpoint_every=2)
    assert len(calls) == 4

    # re-running with the same images should skip all of them (already labeled)
    label_images(image_paths, backend, SYSTEM_PROMPT, USER_PROMPT, 0.1, 512, labels_raw_path, checkpoint_every=2)
    assert len(calls) == 4  # unchanged

    reloaded = load_labels_raw(labels_raw_path)
    assert len(reloaded) == 4


def test_label_images_survives_backend_exceptions(tmp_path):
    image_paths = _make_images(tmp_path, n=3)
    labels_raw_path = tmp_path / "labels_raw.json"

    def flaky_responder(image_path):
        if image_path.stem.endswith("1"):
            raise RuntimeError("simulated backend failure")
        return "[]"

    backend = MockBackend(responder=flaky_responder)
    labels_raw = label_images(image_paths, backend, SYSTEM_PROMPT, USER_PROMPT, 0.1, 512, labels_raw_path)

    assert len(labels_raw) == 3
    assert labels_raw["hard_hat_workers1"]["detections"] == []
    assert "error" in labels_raw["hard_hat_workers1"]
    assert "error" not in labels_raw["hard_hat_workers0"]


def test_resume_retries_only_previously_failed_images(tmp_path):
    """A backend failure must be retried on resume; a genuine empty result must not."""
    image_paths = _make_images(tmp_path, n=3)
    labels_raw_path = tmp_path / "labels_raw.json"

    calls = []

    def flaky_once(image_path):
        calls.append(image_path.stem)
        if image_path.stem.endswith("1"):
            raise RuntimeError("simulated transient backend failure")
        return "[]"

    backend = MockBackend(responder=flaky_once)
    label_images(image_paths, backend, SYSTEM_PROMPT, USER_PROMPT, 0.1, 512, labels_raw_path)
    assert calls == ["hard_hat_workers0", "hard_hat_workers1", "hard_hat_workers2"]

    # second run: backend now succeeds for everyone; only the previously
    # failed image (workers1) should be re-sent, not workers0/workers2
    calls.clear()

    def always_succeeds(image_path):
        calls.append(image_path.stem)
        return "[]"

    backend2 = MockBackend(responder=always_succeeds)
    labels_raw = label_images(image_paths, backend2, SYSTEM_PROMPT, USER_PROMPT, 0.1, 512, labels_raw_path)

    assert calls == ["hard_hat_workers1"]
    assert "error" not in labels_raw["hard_hat_workers1"]


class WedgedThenRecoveredBackend(VLMBackend):
    """Simulates a hung llama-server: every generate() call fails until restart()
    is called, mirroring the real failure observed on a long labeling run where
    /health stayed responsive but /v1/chat/completions requests hung forever."""

    def __init__(self):
        self.healthy = False
        self.restart_count = 0
        self.generate_calls = 0

    def generate(self, image_path, system_prompt, user_prompt, temperature, max_tokens):
        self.generate_calls += 1
        if not self.healthy:
            raise TimeoutError("simulated wedged server")
        return "[]"

    def restart(self):
        self.restart_count += 1
        self.healthy = True


def test_label_images_self_heals_after_consecutive_failures(tmp_path):
    image_paths = _make_images(tmp_path, n=5)
    labels_raw_path = tmp_path / "labels_raw.json"

    backend = WedgedThenRecoveredBackend()
    labels_raw = label_images(
        image_paths,
        backend,
        SYSTEM_PROMPT,
        USER_PROMPT,
        0.1,
        512,
        labels_raw_path,
        max_consecutive_failures=3,
    )

    assert backend.restart_count == 1
    # first 3 images fail before the restart kicks in; the 3rd is retried
    # immediately post-restart and succeeds, and everything after is healthy
    assert "error" in labels_raw["hard_hat_workers0"]
    assert "error" in labels_raw["hard_hat_workers1"]
    assert "error" not in labels_raw["hard_hat_workers2"]
    assert "error" not in labels_raw["hard_hat_workers3"]
    assert "error" not in labels_raw["hard_hat_workers4"]
