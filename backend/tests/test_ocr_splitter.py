"""Unit tests for the OCR splitters.

Covers the four cases requested in the column-splitter spec:
  (a) single line, no gaps to compute -> unchanged
  (b) two columns separated by one large gap -> 2 regions
  (c) five columns separated by four large gaps -> 5 regions
  (d) word spacing within normal range -> no false splits

Plus one case for the EasyOCR over-wide splitter:
  (e) a synthetic 30:1 aspect-ratio region splits into ~15:1 chunks.

Run from project root:
    cd ~/pixelscript/backend
    venv/bin/python -m pytest tests/test_ocr_splitter.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr import _split_line_at_column_gaps


def _word(text, x_min, x_max, y_min=0, y_max=20):
    return {
        "text": text,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


# ---------------------------------------------------------------------------
# Column splitter (_split_line_at_column_gaps)
# ---------------------------------------------------------------------------

def test_single_word_line_unchanged():
    """Case (a): a one-word line has no gaps -> returned as-is."""
    line = [_word("Hello", 0, 50)]
    result = _split_line_at_column_gaps(line)
    assert len(result) == 1
    assert result[0] == line


def test_two_columns_split_at_single_large_gap():
    """Case (b): two columns 'Foo Bar | Baz Qux' split at the wide gap."""
    line = [
        _word("Foo", 0,   30),
        _word("Bar", 35,  65),    # gap 5 (normal)
        _word("Baz", 200, 230),   # gap 135 (column boundary)
        _word("Qux", 235, 265),   # gap 5 (normal)
    ]
    # gaps=[5,135,5], median=5, threshold=max(15, 15)=15, gap 135 > 15 -> split
    result = _split_line_at_column_gaps(line)
    assert len(result) == 2, f"expected 2 groups, got {len(result)}"
    assert [w["text"] for w in result[0]] == ["Foo", "Bar"]
    assert [w["text"] for w in result[1]] == ["Baz", "Qux"]


def test_five_columns_split_at_four_large_gaps():
    """Case (c): five columns, each containing two words, separated by wide
    inter-column gaps -> 5 region groups."""
    line = []
    for col in range(5):
        x_base = col * 200
        line.append(_word("Role", x_base,       x_base + 30))
        line.append(_word(chr(65 + col), x_base + 35, x_base + 55))
    # gaps: [5, 145, 5, 145, 5, 145, 5, 145, 5]
    # median=5, median_word_width=~25, threshold=max(15, 12.5)=15
    # four 145-gaps exceed -> 4 splits -> 5 groups
    result = _split_line_at_column_gaps(line)
    assert len(result) == 5, f"expected 5 groups, got {len(result)}"
    for i, group in enumerate(result):
        assert len(group) == 2, f"group {i} has {len(group)} words, expected 2"
        assert group[0]["text"] == "Role"
        assert group[1]["text"] == chr(65 + i)


def test_uniform_spacing_no_false_splits():
    """Case (d): a normal sentence with uniform word spacing -> no split."""
    line = [
        _word("the",   0,   25),
        _word("quick", 30,  70),
        _word("brown", 75,  115),
        _word("fox",   120, 150),
        _word("jumps", 155, 195),
    ]
    # gaps all = 5, median_width=~37, threshold=max(15, 18.5)=18.5
    # no gap exceeds -> 1 group of 5 words
    result = _split_line_at_column_gaps(line)
    assert len(result) == 1
    assert len(result[0]) == 5


# NOTE: test_over_wide_region_splits_30_to_1 (case e) was removed when its
# subject — the EasyOCR `_split_over_wide_region` helper — was deleted as part
# of replacing EasyOCR with RapidOCR. RapidOCR returns per-line bboxes that
# don't suffer the over-wide-strip pathology that helper was guarding against.
