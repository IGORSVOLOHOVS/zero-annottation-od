from zeod.labeling.grid import FACTOR, rescale_bbox_to_pixels, smart_resize


def test_smart_resize_dimensions_are_multiples_of_factor():
    w, h = smart_resize(width=416, height=416, min_pixels=1024 * FACTOR**2, max_pixels=16384 * FACTOR**2)
    assert w % FACTOR == 0
    assert h % FACTOR == 0


def test_smart_resize_upscales_small_images_to_meet_min_pixels():
    min_pixels = 1024 * FACTOR**2
    w, h = smart_resize(width=416, height=416, min_pixels=min_pixels, max_pixels=16384 * FACTOR**2)
    assert w * h >= min_pixels


def test_smart_resize_downscales_large_images_to_respect_max_pixels():
    max_pixels = 256 * FACTOR**2
    w, h = smart_resize(width=4000, height=3000, min_pixels=4 * FACTOR**2, max_pixels=max_pixels)
    assert w * h <= max_pixels


def test_smart_resize_roughly_preserves_aspect_ratio():
    w, h = smart_resize(width=800, height=400, min_pixels=1024 * FACTOR**2, max_pixels=16384 * FACTOR**2)
    assert abs((w / h) - 2.0) < 0.15


def test_rescale_bbox_to_pixels_identity_when_grid_matches_original():
    bbox = [10, 20, 100, 200]
    result = rescale_bbox_to_pixels(bbox, orig_w=416, orig_h=416, grid_w=416, grid_h=416)
    assert result == bbox


def test_rescale_bbox_to_pixels_scales_down_from_larger_grid():
    # model saw a 2x upscaled canvas; a box spanning the whole grid should
    # rescale back to spanning the whole original image
    bbox = [0, 0, 896, 896]
    result = rescale_bbox_to_pixels(bbox, orig_w=448, orig_h=448, grid_w=896, grid_h=896)
    assert result == [0, 0, 448, 448]


def test_rescale_bbox_to_pixels_handles_non_square_grid():
    bbox = [0, 0, 924, 896]
    result = rescale_bbox_to_pixels(bbox, orig_w=416, orig_h=415, grid_w=924, grid_h=896)
    assert result[2] == 416  # x_max maps back to full original width
    assert result[3] == 415  # y_max maps back to full original height
