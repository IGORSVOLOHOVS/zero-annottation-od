from zeod.dataset.yolo_convert import convert_to_yolo

CLASS_MAP = {"helmet": 0, "no_helmet": 1}


def test_basic_conversion_centers_and_normalizes():
    detections = [{"label": "helmet", "bbox": [100, 100, 200, 300]}]
    lines = convert_to_yolo(detections, img_w=1000, img_h=1000, class_map=CLASS_MAP)

    assert len(lines) == 1
    cls_id, cx, cy, w, h = lines[0].split()
    assert cls_id == "0"
    assert abs(float(cx) - 0.15) < 1e-6
    assert abs(float(cy) - 0.2) < 1e-6
    assert abs(float(w) - 0.1) < 1e-6
    assert abs(float(h) - 0.2) < 1e-6


def test_out_of_bounds_bbox_is_clamped_not_dropped():
    # bbox extends past the top-left corner but still has positive area inside the image
    detections = [{"label": "helmet", "bbox": [-20, -20, 60, 60]}]
    lines = convert_to_yolo(detections, img_w=100, img_h=100, class_map=CLASS_MAP)

    assert len(lines) == 1
    _, cx, cy, w, h = lines[0].split()
    # clamped to [0, 0, 60, 60] -> centered at (0.3, 0.3) with width/height 0.6
    assert abs(float(cx) - 0.3) < 1e-6
    assert abs(float(cy) - 0.3) < 1e-6
    assert abs(float(w) - 0.6) < 1e-6
    assert abs(float(h) - 0.6) < 1e-6


def test_unknown_label_is_dropped():
    detections = [{"label": "hat", "bbox": [0, 0, 10, 10]}]
    assert convert_to_yolo(detections, 100, 100, CLASS_MAP) == []


def test_malformed_bbox_is_dropped():
    for bad_bbox in ([1, 2, 3], "not-a-list", [1, 2, 3, "x"], None):
        detections = [{"label": "helmet", "bbox": bad_bbox}]
        assert convert_to_yolo(detections, 100, 100, CLASS_MAP) == []


def test_degenerate_box_after_clamping_is_dropped():
    # entirely outside the image -> clamps to zero area
    detections = [{"label": "helmet", "bbox": [-50, -50, -10, -10]}]
    assert convert_to_yolo(detections, 100, 100, CLASS_MAP) == []

    # x_max <= x_min even before clamping
    detections = [{"label": "helmet", "bbox": [50, 50, 40, 90]}]
    assert convert_to_yolo(detections, 100, 100, CLASS_MAP) == []


def test_tiny_box_below_min_frac_is_dropped():
    detections = [{"label": "helmet", "bbox": [0, 0, 1, 1]}]
    lines = convert_to_yolo(detections, img_w=1000, img_h=1000, class_map=CLASS_MAP, min_box_frac=0.005)
    assert lines == []


def test_huge_box_above_max_frac_is_dropped():
    detections = [{"label": "helmet", "bbox": [0, 0, 999, 999]}]
    lines = convert_to_yolo(detections, img_w=1000, img_h=1000, class_map=CLASS_MAP, max_box_frac=0.95)
    assert lines == []


def test_label_is_case_and_whitespace_insensitive():
    detections = [{"label": "  HELMET ", "bbox": [10, 10, 50, 50]}]
    lines = convert_to_yolo(detections, 100, 100, CLASS_MAP)
    assert len(lines) == 1
    assert lines[0].startswith("0 ")


def test_empty_detections_returns_empty_list():
    assert convert_to_yolo([], 100, 100, CLASS_MAP) == []


def test_multiple_detections_keep_only_valid_ones():
    detections = [
        {"label": "helmet", "bbox": [10, 10, 50, 50]},
        {"label": "unknown", "bbox": [0, 0, 10, 10]},
        {"label": "no_helmet", "bbox": [60, 60, 90, 90]},
    ]
    lines = convert_to_yolo(detections, 100, 100, CLASS_MAP)
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("1 ")


def test_near_duplicate_same_label_boxes_are_deduped():
    detections = [
        {"label": "helmet", "bbox": [100, 100, 200, 200]},
        {"label": "helmet", "bbox": [102, 101, 199, 202]},  # near-identical repeat
    ]
    lines = convert_to_yolo(detections, 1000, 1000, CLASS_MAP, dedup_iou_threshold=0.9)
    assert len(lines) == 1


def test_near_duplicate_different_label_boxes_are_not_deduped():
    detections = [
        {"label": "helmet", "bbox": [100, 100, 200, 200]},
        {"label": "no_helmet", "bbox": [100, 100, 200, 200]},
    ]
    lines = convert_to_yolo(detections, 1000, 1000, CLASS_MAP, dedup_iou_threshold=0.9)
    assert len(lines) == 2


def test_dedup_disabled_keeps_both_boxes():
    detections = [
        {"label": "helmet", "bbox": [100, 100, 200, 200]},
        {"label": "helmet", "bbox": [100, 100, 200, 200]},
    ]
    lines = convert_to_yolo(detections, 1000, 1000, CLASS_MAP, dedup_iou_threshold=None)
    assert len(lines) == 2


def test_head_crop_squares_a_tall_box_anchored_at_top():
    # width 40, height 200 -> h/w = 5.0, well above the 1.8 threshold
    detections = [{"label": "no_helmet", "bbox": [100, 50, 140, 250]}]
    lines = convert_to_yolo(detections, 1000, 1000, CLASS_MAP, head_crop_max_aspect_ratio=1.8)
    assert len(lines) == 1
    _, cx, cy, w, h = lines[0].split()
    # cropped box should be [100, 50, 140, 90] -> square, width==height==40
    assert abs(float(w) * 1000 - 40) < 1e-6
    assert abs(float(h) * 1000 - 40) < 1e-6
    assert abs(float(cy) * 1000 - 70) < 1e-6  # (50+90)/2


def test_head_crop_leaves_already_square_box_unchanged():
    detections = [{"label": "helmet", "bbox": [100, 100, 150, 150]}]  # h/w = 1.0
    lines = convert_to_yolo(detections, 1000, 1000, CLASS_MAP, head_crop_max_aspect_ratio=1.8)
    assert len(lines) == 1
    _, cx, cy, w, h = lines[0].split()
    assert abs(float(w) - float(h)) < 1e-9


def test_head_crop_disabled_by_default_in_function_signature():
    # a tall box should pass through unchanged when head_crop_max_aspect_ratio is not passed
    detections = [{"label": "no_helmet", "bbox": [100, 50, 140, 250]}]
    lines = convert_to_yolo(detections, 1000, 1000, CLASS_MAP)
    _, cx, cy, w, h = lines[0].split()
    assert abs(float(h) * 1000 - 200) < 1e-6  # unchanged height
