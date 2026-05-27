"""Tests for the RapidOCR raw-output -> region schema mapping.

These tests don't require rapidocr-onnxruntime to be installed: they exercise
the pure mapping function _rapidocr_results_to_regions with synthetic raw
output. This keeps the tests deterministic (no model variability), fast (no
OCR inference), and runnable in any environment.

Run from project root:
    cd ~/pixelscript/backend
    venv/bin/python -m pytest tests/test_rapidocr_mapping.py -v
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from ocr import _rapidocr_results_to_regions, CONFIDENCE_THRESHOLD


def _polygon(x1, y1, x2, y2):
    """RapidOCR returns 4-point polygons in top-left, top-right, bottom-right,
    bottom-left order. Helper to build them from x/y ranges."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_empty_input_returns_empty():
    assert _rapidocr_results_to_regions([]) == []


def test_single_line_two_words_maps_to_one_region():
    raw = [[_polygon(10, 20, 200, 50), "hello world", 0.95]]
    regions = _rapidocr_results_to_regions(raw)
    assert len(regions) == 1
    r = regions[0]
    assert r["text"] == "hello world"
    assert r["confidence"] == 0.95
    assert r["bounding_box"] == {"x": 10, "y": 20, "width": 190, "height": 30}
    assert len(r["_words"]) == 2
    assert r["_words"][0]["text"] == "hello"
    assert r["_words"][1]["text"] == "world"


def test_region_schema_keys_match_other_backends():
    """Output shape must match Vision/Tesseract producers exactly."""
    raw = [[_polygon(0, 0, 100, 30), "sample", 0.9]]
    regions = _rapidocr_results_to_regions(raw)
    r = regions[0]
    assert set(r.keys()) == {"text", "confidence", "bounding_box", "_words"}
    assert set(r["bounding_box"].keys()) == {"x", "y", "width", "height"}
    for v in r["bounding_box"].values():
        assert isinstance(v, int)


def test_words_widths_tile_line_bbox_approximately():
    """Per-word bboxes should fit edge-to-edge within the line bbox, with
    widths proportional to char count."""
    raw = [[_polygon(0, 0, 300, 30), "alpha beta gamma", 0.9]]
    regions = _rapidocr_results_to_regions(raw)
    words = regions[0]["_words"]
    assert [w["text"] for w in words] == ["alpha", "beta", "gamma"]
    # char widths: 5, 4, 5 -> total 14 -> proportions 5/14, 4/14, 5/14
    # line width 300 -> word widths ~107, 86, 107 (subject to rounding)
    total_w = sum(w["x_max"] - w["x_min"] for w in words)
    assert abs(total_w - 300) <= 2
    # Words tile left-to-right without gaps
    assert words[0]["x_min"] == 0
    assert words[-1]["x_max"] in (299, 300, 301)  # rounding tolerance


def test_low_confidence_regions_filtered():
    raw = [
        [_polygon(0, 0, 100, 30), "kept", 0.95],
        [_polygon(0, 40, 100, 70), "dropped", 0.1],  # below CONFIDENCE_THRESHOLD
    ]
    regions = _rapidocr_results_to_regions(raw)
    assert len(regions) == 1
    assert regions[0]["text"] == "kept"
    # Sanity: the dropped region's confidence is indeed below threshold
    assert 0.1 < CONFIDENCE_THRESHOLD


def test_empty_text_regions_filtered():
    raw = [[_polygon(0, 0, 100, 30), "   ", 0.99]]
    assert _rapidocr_results_to_regions(raw) == []


def test_degenerate_bbox_filtered():
    """Zero-area polygons (e.g., corrupt detections) must not produce
    regions — they'd crash font_replacer downstream."""
    raw = [[_polygon(10, 20, 10, 20), "zero-area", 0.95]]
    assert _rapidocr_results_to_regions(raw) == []


def test_single_word_line_has_one_word_meta():
    raw = [[_polygon(0, 0, 100, 30), "solo", 0.9]]
    regions = _rapidocr_results_to_regions(raw)
    assert len(regions[0]["_words"]) == 1
    assert regions[0]["_words"][0]["text"] == "solo"
    assert regions[0]["_words"][0]["x_min"] == 0
