"""Post-OCR font-size normalization across visually-aligned region groups.

Groups regions by (same visual row AND similar bbox height) and shrinks each
region's bbox so they render at a shared target size = min(predicted_sizes
in group), floored at the same legibility threshold the validator uses.

Order in the pipeline:
    detect_text -> validate_legibility -> normalize_fontsize_groups -> replace_fonts

The validator guarantees every region renders above the legibility threshold;
the normalizer makes visually-aligned regions render at consistent sizes,
without ever dropping back below that threshold.

Sizing rules are NOT duplicated from font_replacer. The normalizer binary-
searches font_replacer.find_font_size to find the bbox height that yields a
target size — single source of truth, no drift if multipliers ever change.
"""
import logging
from PIL import Image, ImageDraw

from font_replacer import find_font_size, get_font_path, BOLD_FONTS
from region_validator import BODY_MIN_PX, HEADING_MIN_PX, HEADING_PERCENTILE

logger = logging.getLogger(__name__)

Y_OVERLAP_FRACTION = 0.5        # 50% of smaller height -> "same row"
HEIGHT_SIMILARITY_RATIO = 0.40  # within +/-40% bbox height -> typographically related


def normalize_fontsize_groups(regions, font_name):
    """Shrink bbox heights so visually-aligned regions render at a shared size.

    Conservative: only shrinks (never expands — expansion could overlap
    neighbors). Floored at legibility thresholds so the validator's
    legibility guarantee is preserved.
    """
    if not regions or len(regions) < 2:
        return regions

    font_path = get_font_path(font_name)
    is_bold = font_name in BOLD_FONTS

    # Same heading classification the validator uses.
    heights = sorted(r["bounding_box"]["height"] for r in regions)
    cutoff_idx = min(int(len(heights) * HEADING_PERCENTILE), len(heights) - 1)
    heading_cutoff = heights[cutoff_idx]

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    out = []
    for group in _group_by_visual_row(regions):
        if len(group) < 2:
            out.extend(group)
            continue
        predicted = [
            find_font_size(draw, r["text"], font_path, r["bounding_box"], is_bold).size
            for r in group
        ]
        has_heading = any(r["bounding_box"]["height"] >= heading_cutoff for r in group)
        floor = HEADING_MIN_PX if has_heading else BODY_MIN_PX
        target_size = max(min(predicted), floor)
        for r in group:
            out.append(_apply_target_size(r, target_size, is_bold, draw, font_path))
    return out


def _group_by_visual_row(regions):
    """Union-find: regions are grouped iff y-overlap >= 50% of smaller height
    AND bbox heights within +/-40%. Transitive."""
    n = len(regions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        bi = regions[i]["bounding_box"]
        yi1, yi2, hi = bi["y"], bi["y"] + bi["height"], bi["height"]
        for j in range(i + 1, n):
            bj = regions[j]["bounding_box"]
            yj1, yj2, hj = bj["y"], bj["y"] + bj["height"], bj["height"]
            overlap = max(0, min(yi2, yj2) - max(yi1, yj1))
            min_h = min(hi, hj)
            if min_h == 0 or overlap < Y_OVERLAP_FRACTION * min_h:
                continue
            if min(hi, hj) < (1 - HEIGHT_SIMILARITY_RATIO) * max(hi, hj):
                continue
            union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(regions[i])
    return list(groups.values())


def _apply_target_size(region, target_size, is_bold, draw, font_path):
    """Shrink region's bbox height to whatever value makes find_font_size
    yield <= target_size. Only shrinks (no expansion). Centers vertically."""
    bbox = region["bounding_box"]
    target_h = _bbox_height_for_target(
        draw, region["text"], font_path, bbox, is_bold, target_size
    )
    if target_h >= bbox["height"]:
        return region  # already at or below target

    new_y = bbox["y"] + (bbox["height"] - target_h) // 2
    out = dict(region)
    out["bounding_box"] = {
        "x": bbox["x"],
        "y": new_y,
        "width": bbox["width"],
        "height": target_h,
    }
    return out


def _bbox_height_for_target(draw, text, font_path, bbox, is_bold, target_size):
    """Binary search for the largest bbox height that makes find_font_size
    return <= target_size. Reuses find_font_size as the source of truth so
    sizing rules can never drift between font_replacer and this module."""
    max_h = bbox["height"]
    if max_h <= 1:
        return max_h
    lo, hi = 1, max_h
    best = max_h
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = dict(bbox)
        candidate["height"] = mid
        size = find_font_size(draw, text, font_path, candidate, is_bold).size
        if size <= target_size:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best
