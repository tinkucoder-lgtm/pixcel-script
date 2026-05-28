"""Regression guard for font_replacer.py.

This file has been silently rewritten twice. The tests below pin the locked
spec so a third rewrite fails CI loudly rather than silently breaking output.

Run from project root:
    cd ~/pixelscript/backend
    venv/bin/python -m pytest tests/test_font_replacer_spec.py -v
"""
from pathlib import Path

FONT_REPLACER = Path(__file__).resolve().parent.parent / "font_replacer.py"


def _source() -> str:
    return FONT_REPLACER.read_text(encoding="utf-8")


def test_file_exists():
    assert FONT_REPLACER.exists(), f"font_replacer.py missing at {FONT_REPLACER}"


def test_line_count_within_tolerance():
    lines = _source().splitlines()
    assert 94 <= len(lines) <= 104, (
        f"font_replacer.py is {len(lines)} lines — outside locked range 94-104. "
        "Has it been rewritten? See project status doc."
    )


def test_char_bucket_height_multipliers():
    """1-2 chars -> 0.45h, 3-6 -> 0.60h, 7-20 -> 0.90h, 20+ -> 0.85h."""
    src = _source()
    for marker in ("0.45", "0.60", "0.90", "0.85"):
        assert marker in src, f"Missing height multiplier {marker}"


def test_char_bucket_width_limits():
    """3-6 -> 0.95w, 7-20 -> 0.92w, 20+ -> 0.88w, bold -> 0.80w."""
    src = _source()
    for marker in ("0.95", "0.92", "0.88", "0.80"):
        assert marker in src, f"Missing width limit {marker}"


def test_char_bucket_boundaries():
    src = _source()
    for boundary in ("n <= 2", "n <= 6", "n <= 20"):
        assert boundary in src, f"Char bucket boundary '{boundary}' missing"


def test_bold_scale_factor():
    assert "0.75" in _source(), "Bold fonts must apply x0.75 start factor"


def test_bold_fonts_complete():
    src = _source()
    for name in ("bebas-neue", "abril-fatface", "oswald", "lobster", "righteous"):
        assert f'"{name}"' in src, f"BOLD_FONTS missing '{name}'"


def test_inpaint_algorithm_and_radius():
    src = _source()
    assert "INPAINT_TELEA" in src, "Inpaint algorithm must be INPAINT_TELEA"
    assert "inpaintRadius=9" in src, "Inpaint radius must be 9"


def test_inpaint_mask_pad_is_8():
    """Mask is padded by 8 pixels: x-8, y-8, +w+8, +h+8.

    Increased from 2 to 8 to reduce text ghosting on photographic backgrounds —
    larger pad means the inpaint algorithm has more clean pixels to sample
    from when reconstructing the masked-out area.
    """
    src = _source()
    assert "x-8" in src and "y-8" in src, "Mask must use 8px pad (x-8, y-8)"
    assert "w+8" in src and "h+8" in src, "Mask must use 8px pad (w+8, h+8)"


def test_color_sampling_strip():
    assert "strip = 6" in _source(), "Color sample strip must be 6 pixels"


def test_brightness_thresholds():
    src = _source()
    assert "> 200" in src, "Brightness >200 threshold (black) missing"
    assert "> 140" in src, "Brightness >140 threshold (dark grey) missing"


def test_brightness_color_outputs():
    src = _source()
    assert "(0, 0, 0)" in src, "Black output (0,0,0) missing"
    assert "(30, 30, 30)" in src, "Dark grey output (30,30,30) missing"
    assert "(255, 255, 255)" in src, "White output (255,255,255) missing"
