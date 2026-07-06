import yaml
from PIL import Image

from zeod.dataset.build import build_dataset

CLASS_MAP = {"helmet": 0, "no_helmet": 1}


def _make_image(path, size=(200, 150)):
    Image.new("RGB", size, color=(128, 128, 128)).save(path)
    return path


def _labels_raw(tmp_path, n=8):
    labels_raw = {}
    for i in range(n):
        img_path = _make_image(tmp_path / f"img_{i}.png")
        detections = [{"label": "helmet" if i % 2 == 0 else "no_helmet", "bbox": [10, 10, 60, 60]}]
        labels_raw[f"img_{i}"] = {"image_path": str(img_path), "raw_response": "...", "detections": detections}
    # one image with no detections - should be skipped entirely
    empty_path = _make_image(tmp_path / "img_empty.png")
    labels_raw["img_empty"] = {"image_path": str(empty_path), "raw_response": "[]", "detections": []}
    return labels_raw


def test_build_dataset_produces_valid_yolo_labels(tmp_path):
    labels_raw = _labels_raw(tmp_path, n=8)
    dataset_dir = tmp_path / "dataset"

    stats = build_dataset(
        labels_raw=labels_raw,
        dataset_dir=dataset_dir,
        class_map=CLASS_MAP,
        val_split=0.25,
        min_box_frac=0.005,
        max_box_frac=0.95,
        seed=42,
    )

    assert stats.n_valid == 8
    assert stats.n_skipped == 1  # img_empty
    assert stats.n_train + stats.n_val == 8

    for split in ("train", "val"):
        images = sorted((dataset_dir / split / "images").glob("*.png"))
        labels = sorted((dataset_dir / split / "labels").glob("*.txt"))
        assert len(images) == len(labels)
        assert len(images) > 0

        for label_file in labels:
            lines = label_file.read_text(encoding="utf-8").splitlines()
            assert len(lines) >= 1
            for line in lines:
                parts = line.split()
                assert len(parts) == 5
                cls_id = int(parts[0])
                assert cls_id in CLASS_MAP.values()
                cx, cy, w, h = (float(v) for v in parts[1:])
                for value in (cx, cy, w, h):
                    assert 0.0 <= value <= 1.0


def test_build_dataset_writes_valid_data_yaml(tmp_path):
    labels_raw = _labels_raw(tmp_path, n=6)
    dataset_dir = tmp_path / "dataset"

    build_dataset(labels_raw, dataset_dir, CLASS_MAP, val_split=0.2, min_box_frac=0.005, max_box_frac=0.95, seed=1)

    data_yaml = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["train"] == "train/images"
    assert data_yaml["val"] == "val/images"
    assert data_yaml["names"] == {0: "helmet", 1: "no_helmet"}


def test_build_dataset_handles_no_valid_annotations(tmp_path):
    img_path = _make_image(tmp_path / "img_0.png")
    labels_raw = {"img_0": {"image_path": str(img_path), "raw_response": "[]", "detections": []}}
    dataset_dir = tmp_path / "dataset"

    stats = build_dataset(
        labels_raw, dataset_dir, CLASS_MAP, val_split=0.2, min_box_frac=0.005, max_box_frac=0.95, seed=1
    )

    assert stats.n_valid == 0
    assert not (dataset_dir / "data.yaml").exists()
