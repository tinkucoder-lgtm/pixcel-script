"""Tests for the post-OCR legibility validator."""
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from region_validator import validate_legibility, _split_region


def _available_font_name():
    fonts_dir = BACKEND / "fonts"
    ttfs = sorted(fonts_dir.glob("*.ttf"))
    if not ttfs:
        pytest.skip("No .ttf fonts in backend/fonts/")
    return ttfs[0].stem


def test_no_op_on_legible_region():
    """A short region with a generous bbox renders legibly => unchanged."""
    font_name = _available_font_name()
    region = {
        "text": "Hello",
        "confidence": 1.0,
        "bounding_box": {"x": 0, "y": 0, "width": 300, "height": 80},
    }
    result = validate_legibility([region], font_name)
    assert len(result) == 1
    assert result[0]["text"] == "Hello"
    assert result[0]["bounding_box"] == region["bounding_box"]


def test_splits_over_merged_mesa_row():
    """A wide-but-short region (Mesa role-titles smushed) gets split."""
    font_name = _available_font_name()
    region = {
        "text": "FOUNDER CHIEF OF STAFF PRODUCT MANAGER GROWTH LEAD FOUNDERS OFFICE",
        "confidence": 1.0,
        "bounding_box": {"x": 0, "y": 100, "width": 800, "height": 25},
        # Intentionally no _words: exercises char-proportional fallback path.
    }
    result = validate_legibility([region], font_name)
    assert len(result) >= 2, f"expected at least 2 sub-regions, got {len(result)}"
    for r in result:
        assert r["text"].strip(), "every sub-region must carry text"
        assert r["bounding_box"]["width"] > 0


def test_unsplittable_single_word_accepted_not_looped():
    """A single long word in a tiny bbox => accepted illegible, no loop."""
    font_name = _available_font_name()
    region = {
        "text": "Antidisestablishmentarianism",
        "confidence": 1.0,
        "bounding_box": {"x": 0, "y": 0, "width": 50, "height": 10},
    }
    result = validate_legibility([region], font_name, max_iterations=20)
    assert len(result) == 1
    assert result[0]["text"] == region["text"]


def test_tall_bbox_classified_as_heading_passes_through():
    """A region in the top 25% of bbox heights is treated as heading; if its
    predicted size comfortably exceeds 18px, it passes through unchanged
    even when many smaller regions are present."""
    font_name = _available_font_name()
    body_regions = [
        {
            "text": "body region",
            "confidence": 1.0,
            "bounding_box": {"x": i * 200, "y": 0, "width": 200, "height": 20},
        }
        for i in range(3)
    ]
    heading_region = {
        "text": "TITLE",
        "confidence": 1.0,
        "bounding_box": {"x": 0, "y": 200, "width": 500, "height": 100},
    }
    result = validate_legibility(body_regions + [heading_region], font_name)
    surviving_titles = [r for r in result if r["text"] == "TITLE"]
    assert len(surviving_titles) == 1
    assert surviving_titles[0]["bounding_box"]["height"] == 100


def test_split_region_uses_words_when_available():
    """_split_region produces tighter, position-accurate bboxes when _words
    metadata is present; falls back to char-proportional division when not."""
    region_with_words = {
        "text": "alpha beta gamma delta",
        "confidence": 1.0,
        "bounding_box": {"x": 0, "y": 0, "width": 1000, "height": 30},
        "_words": [
            {"text": "alpha", "x_min": 0,   "x_max": 50,   "y_min": 0, "y_max": 30},
            {"text": "beta",  "x_min": 60,  "x_max": 100,  "y_min": 0, "y_max": 30},
            {"text": "gamma", "x_min": 800, "x_max": 870,  "y_min": 0, "y_max": 30},
            {"text": "delta", "x_min": 880, "x_max": 1000, "y_min": 0, "y_max": 30},
        ],
    }
    region_no_words = {
        "text": "alpha beta gamma delta",
        "confidence": 1.0,
        "bounding_box": {"x": 0, "y": 0, "width": 1000, "height": 30},
    }
    sub_with = _split_region(region_with_words)
    sub_without = _split_region(region_no_words)

    assert len(sub_with) == 2
    assert len(sub_without) == 2

    # _words path: tight bboxes matching actual word positions
    assert sub_with[0]["bounding_box"]["x"] == 0
    assert sub_with[0]["bounding_box"]["width"] == 100   # alpha+beta span 0-100
    assert sub_with[1]["bounding_box"]["x"] == 800
    assert sub_with[1]["bounding_box"]["width"] == 200   # gamma+delta span 800-1000

    # Char-proportional path: bbox split by character count, ignoring real positions
    # Left "alpha beta" = 10 chars, right "gamma delta" = 11 chars, total 21
    # left_w = int(1000 * 10/21) = 476
    assert sub_without[0]["bounding_box"]["x"] == 0
    assert sub_without[0]["bounding_box"]["width"] == 476
    assert sub_without[1]["bounding_box"]["x"] == 476
    assert sub_without[1]["bounding_box"]["width"] == 524
