from zeod.labeling.parser import parse_json_response


def test_clean_json_array():
    text = '[{"label": "helmet", "bbox": [1, 2, 3, 4]}]'
    result = parse_json_response(text)
    assert result == [{"label": "helmet", "bbox": [1, 2, 3, 4]}]


def test_empty_array():
    assert parse_json_response("[]") == []


def test_empty_string():
    assert parse_json_response("") == []


def test_whitespace_only():
    assert parse_json_response("   \n\t  ") == []


def test_markdown_code_fence():
    text = '```json\n[{"label": "no_helmet", "bbox": [0, 0, 10, 10]}]\n```'
    result = parse_json_response(text)
    assert result == [{"label": "no_helmet", "bbox": [0, 0, 10, 10]}]


def test_prose_before_and_after_json():
    text = (
        'Sure! Here is the detection result:\n[{"label": "helmet", "bbox": [5, 5, 15, 15]}]'
        "\nLet me know if you need more."
    )
    result = parse_json_response(text)
    assert result == [{"label": "helmet", "bbox": [5, 5, 15, 15]}]


def test_malformed_json_returns_empty():
    assert parse_json_response('[{"label": "helmet", "bbox": [1, 2, 3, 4]') == []


def test_no_array_in_text_returns_empty():
    assert parse_json_response("I could not detect any people.") == []


def test_non_list_json_returns_empty():
    assert parse_json_response('{"label": "helmet"}') == []


def test_non_dict_items_are_dropped():
    result = parse_json_response('[{"label": "helmet"}, "garbage", 42, null]')
    assert result == [{"label": "helmet"}]


def test_truncated_response_returns_empty():
    # simulates hitting max_tokens mid-generation
    text = '[{"label": "helmet", "bbox": [1, 2, 3'
    assert parse_json_response(text) == []


def test_bbox_2d_key_is_normalized_to_bbox():
    # Qwen2.5-VL's own grounding format uses "bbox_2d" regardless of the prompt's requested key
    text = '[{"bbox_2d": [1, 2, 3, 4], "label": "helmet"}]'
    result = parse_json_response(text)
    assert result == [{"bbox_2d": [1, 2, 3, 4], "label": "helmet", "bbox": [1, 2, 3, 4]}]


def test_existing_bbox_key_is_not_overwritten_by_alias():
    text = '[{"bbox": [9, 9, 9, 9], "bbox_2d": [1, 1, 1, 1], "label": "helmet"}]'
    result = parse_json_response(text)
    assert result[0]["bbox"] == [9, 9, 9, 9]
