"""Tests for the RapidOCR two-pass pipeline.

Exercises pure functions (_text_looks_like_missing_spaces,
_translate_pass2_results) with synthetic inputs. No engine, no image, no
network — fully deterministic and runnable without rapidocr installed.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from ocr import (
    _text_looks_like_missing_spaces,
    _translate_pass2_results,
    CONFIDENCE_THRESHOLD,
)


def _polygon(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ---------------------------------------------------------------------------
# Heuristic: _text_looks_like_missing_spaces
# ---------------------------------------------------------------------------

def test_heuristic_catches_punctuation_mixed_phrase():
    """'NOTANMBA,ASTARTUPGARAGE' has letters + comma + length>=8 -> trigger."""
    assert _text_looks_like_missing_spaces("NOTANMBA,ASTARTUPGARAGE") is True


def test_heuristic_catches_phrase_with_colon():
    """'MESAPROGRAMHIGHLIGHTS:' has letters + colon -> trigger."""
    assert _text_looks_like_missing_spaces("MESAPROGRAMHIGHLIGHTS:") is True


def test_heuristic_skips_long_single_word_of_only_letters():
    """'INTERNSHIPS' is a real single word; pass-2 would be wasted work."""
    assert _text_looks_like_missing_spaces("INTERNSHIPS") is False


def test_heuristic_skips_already_spaced_text():
    assert _text_looks_like_missing_spaces("MESA INCUBATOR") is False


def test_heuristic_skips_short_tokens():
    """Short tokens like 'MBA', '100+', 'CXO' are usually genuine, not merges."""
    assert _text_looks_like_missing_spaces("MBA") is False
    assert _text_looks_like_missing_spaces("100+") is False


def test_heuristic_returns_false_on_empty():
    assert _text_looks_like_missing_spaces("") is False


# ---------------------------------------------------------------------------
# Pass-2 translator: _translate_pass2_results
# ---------------------------------------------------------------------------

def test_pass2_sub_bboxes_tile_within_original_and_text_content_preserved():
    """Pass-2 may emit N sub-regions (different count, different bboxes from
    pass-1) — but: sub-bboxes must tile within the original bbox plus crop
    padding, and the concatenated text content must equal the original
    (modulo spaces). This is the machine-verifiable form of the decoupling
    claim: pass-2 is allowed to refine geometry and split regions, but it
    must NOT lose or add character content vs. what pass-1 detected."""
    pass1_region = {
        "text": "NOTANMBA",
        "confidence": 0.9,
        "bounding_box": {"x": 100, "y": 200, "width": 300, "height": 50},
    }
    # Crop was at (96, 196) -> 308x58, so crop_x=96, crop_y=196.
    crop_x, crop_y = 96, 196
    pass2_raw = [
        [_polygon(4,   4, 84,  54), "NOT", 0.95],
        [_polygon(100, 4, 160, 54), "AN",  0.95],
        [_polygon(180, 4, 304, 54), "MBA", 0.95],
    ]
    result = _translate_pass2_results(pass2_raw, pass1_region, crop_x, crop_y)

    # Pass-2 IS allowed to change region count and bboxes (different shape from pass-1).
    assert result is not None
    assert len(result) == 3

    # Text content equivalence — modulo whitespace.
    pass1_no_spaces = pass1_region["text"].replace(" ", "")
    pass2_no_spaces = "".join(r["text"] for r in result).replace(" ", "")
    assert pass1_no_spaces == pass2_no_spaces, (
        f"text content drift: pass-1 had {pass1_no_spaces!r}, "
        f"pass-2 reconstructed {pass2_no_spaces!r}"
    )

    # Sub-bboxes tile within the original bbox (allowing for the 4px crop pad).
    orig = pass1_region["bounding_box"]
    pad = 4
    for r in result:
        rb = r["bounding_box"]
        assert rb["x"] >= orig["x"] - pad
        assert rb["x"] + rb["width"] <= orig["x"] + orig["width"] + pad
        assert rb["y"] >= orig["y"] - pad
        assert rb["y"] + rb["height"] <= orig["y"] + orig["height"] + pad


def test_pass2_translator_returns_none_when_only_one_detection():
    """If pass-2 finds only 1 box (no split), translator returns None so
    the caller keeps the pass-1 region unchanged."""
    pass1_region = {
        "text": "X",
        "confidence": 0.9,
        "bounding_box": {"x": 0, "y": 0, "width": 100, "height": 30},
    }
    pass2_raw = [[_polygon(0, 0, 100, 30), "X", 0.95]]
    assert _translate_pass2_results(pass2_raw, pass1_region, 0, 0) is None


def test_pass2_translator_filters_low_confidence_sub_detections():
    pass1_region = {
        "text": "ABCDEFGH",
        "confidence": 0.9,
        "bounding_box": {"x": 0, "y": 0, "width": 100, "height": 30},
    }
    pass2_raw = [
        [_polygon(0,  0, 30, 30), "ABC",     0.95],
        [_polygon(35, 0, 65, 30), "junk",    0.1],   # below CONFIDENCE_THRESHOLD
        [_polygon(70, 0, 100, 30), "DEF",    0.95],
    ]
    assert 0.1 < CONFIDENCE_THRESHOLD  # sanity
    result = _translate_pass2_results(pass2_raw, pass1_region, 0, 0)
    assert result is not None
    assert len(result) == 2
    assert [r["text"] for r in result] == ["ABC", "DEF"]


def test_pass2_translator_attaches_words_metadata_to_each_sub_region():
    pass1_region = {
        "text": "FOOBAR",
        "confidence": 0.9,
        "bounding_box": {"x": 50, "y": 100, "width": 200, "height": 30},
    }
    pass2_raw = [
        [_polygon(0,   0, 80,  30), "FOO BAR", 0.95],
        [_polygon(100, 0, 200, 30), "BAZ",     0.95],
    ]
    result = _translate_pass2_results(pass2_raw, pass1_region, 50, 100)
    assert result is not None
    # First sub-region has 2 words -> 2 word-meta entries
    assert len(result[0]["_words"]) == 2
    assert [w["text"] for w in result[0]["_words"]] == ["FOO", "BAR"]
    # Second sub-region has 1 word -> 1 word-meta entry
    assert len(result[1]["_words"]) == 1
    assert result[1]["_words"][0]["text"] == "BAZ"
