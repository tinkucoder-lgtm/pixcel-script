"""Tests for the font-size normalizer (region_normalizer.py)."""
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from region_normalizer import (
    normalize_fontsize_groups,
    _group_by_visual_row,
    _bbox_height_for_target,
)
from region_validator import BODY_MIN_PX, HEADING_MIN_PX


def _available_font_name():
    fonts_dir = BACKEND / "fonts"
    ttfs = sorted(fonts_dir.glob("*.ttf"))
    if not ttfs:
        pytest.skip("No .ttf fonts in backend/fonts/")
    return ttfs[0].stem


def _region(text, x, y, w, h):
    return {
        "text": text,
        "confidence": 1.0,
        "bounding_box": {"x": x, "y": y, "width": w, "height": h},
    }


# ---------------------------------------------------------------------------
# Grouping heuristic (_group_by_visual_row)
# ---------------------------------------------------------------------------

def test_grouping_same_row_similar_height_grouped():
    regions = [
        _region("ALPHA", 0,   100, 200, 40),
        _region("BETA",  300, 100, 200, 40),
        _region("GAMMA", 600, 100, 200, 40),
    ]
    groups = _group_by_visual_row(regions)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_grouping_same_y_very_different_heights_not_grouped():
    """Height-similarity guard: a 100px-tall heading next to a 15px-tall
    page number that happens to overlap y must NOT be merged."""
    regions = [
        _region("HEADING", 0,   100, 300, 100),
        _region("12",      400, 130, 30,  15),  # overlaps in y but tiny
    ]
    groups = _group_by_visual_row(regions)
    assert len(groups) == 2


def test_grouping_different_rows_not_grouped():
    regions = [
        _region("ROW1", 0, 0,   200, 30),
        _region("ROW2", 0, 100, 200, 30),
    ]
    groups = _group_by_visual_row(regions)
    assert len(groups) == 2


def test_grouping_is_transitive():
    """A groups with B, B groups with C => A, B, C all in one group."""
    regions = [
        _region("A", 0,   100, 100, 40),
        _region("B", 200, 100, 100, 40),  # overlaps A
        _region("C", 400, 100, 100, 40),  # overlaps B (and A by transitivity)
    ]
    groups = _group_by_visual_row(regions)
    assert len(groups) == 1
    assert len(groups[0]) == 3


# ---------------------------------------------------------------------------
# Inverse find_font_size (_bbox_height_for_target)
# ---------------------------------------------------------------------------

def test_bbox_height_for_target_returns_value_yielding_at_most_target():
    """Source-of-truth property: whatever height the binary search returns
    must, when fed back into find_font_size, produce a size <= target."""
    from PIL import Image, ImageDraw
    from font_replacer import find_font_size, get_font_path
    font_name = _available_font_name()
    font_path = get_font_path(font_name)
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox = {"x": 0, "y": 0, "width": 300, "height": 100}
    target = 20
    h = _bbox_height_for_target(draw, "HELLO WORLD", font_path, bbox, False, target)
    candidate = dict(bbox)
    candidate["height"] = h
    actual = find_font_size(draw, "HELLO WORLD", font_path, candidate, False).size
    assert actual <= target, f"binary search returned h={h}, find_font_size gave {actual}px (> target {target})"


# ---------------------------------------------------------------------------
# End-to-end normalize_fontsize_groups
# ---------------------------------------------------------------------------

def test_normalizer_no_op_on_singleton_input():
    font_name = _available_font_name()
    regions = [_region("Hello", 0, 0, 300, 80)]
    assert normalize_fontsize_groups(regions, font_name) == regions


def test_normalizer_preserves_region_count():
    font_name = _available_font_name()
    regions = [
        _region("ALPHA", 0,   0, 200, 40),
        _region("BETA",  300, 0, 200, 40),
        _region("GAMMA", 0,   200, 100, 30),
    ]
    result = normalize_fontsize_groups(regions, font_name)
    assert len(result) == len(regions)


def test_normalizer_shrinks_only_never_expands():
    """A region that would naturally render at target size or below must NOT
    have its bbox grown by the normalizer."""
    font_name = _available_font_name()
    regions = [
        _region("BIG TITLE TEXT WITH ROOM",  0,   0, 800, 80),
        _region("xs",                         900, 0, 30,  30),
    ]
    result = normalize_fontsize_groups(regions, font_name)
    for orig, new in zip(regions, result):
        assert new["bounding_box"]["height"] <= orig["bounding_box"]["height"]
        assert new["bounding_box"]["width"] == orig["bounding_box"]["width"]
        assert new["bounding_box"]["x"] == orig["bounding_box"]["x"]


def test_normalizer_does_not_break_unrelated_regions():
    """Regions in separate groups should be independent — shrinking one
    group must not touch members of another."""
    font_name = _available_font_name()
    regions = [
        _region("ROW1A", 0,   0,   400, 60),
        _region("ROW1B", 500, 0,   400, 60),
        _region("ROW2",  0,   500, 400, 25),  # different row, different group
    ]
    result = normalize_fontsize_groups(regions, font_name)
    row2_new = next(r for r in result if r["text"] == "ROW2")
    assert row2_new["bounding_box"] == regions[2]["bounding_box"]
