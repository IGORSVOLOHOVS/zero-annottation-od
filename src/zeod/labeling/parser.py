"""Parsing of raw VLM text output into a list of detection dicts."""

from __future__ import annotations

import json
import re

_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

# Qwen2.5-VL was instruction-tuned on its own grounding format and sometimes
# ignores a custom prompt's exact key name, emitting "bbox_2d" instead of the
# requested "bbox". Normalize known aliases so downstream code only ever
# needs to look for "bbox".
_BBOX_KEY_ALIASES = ("bbox", "bbox_2d", "box", "box_2d")


def _normalize_detection(item: dict) -> dict:
    if "bbox" in item:
        return item
    for alias in _BBOX_KEY_ALIASES:
        if alias in item:
            return {**item, "bbox": item[alias]}
    return item


def parse_json_response(text: str) -> list[dict]:
    """Extract a JSON array of detections from a model's free-form text response.

    Handles common VLM quirks: markdown code fences, leading/trailing prose,
    truncated or malformed JSON, and empty responses. Returns [] on any
    failure instead of raising, since a bad response for one image should
    not stop the labeling run.
    """
    if not text:
        return []

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

    match = _ARRAY_RE.search(text)
    if not match:
        return []

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return [_normalize_detection(item) for item in data if isinstance(item, dict)]
