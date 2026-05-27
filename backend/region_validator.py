"""Post-OCR legibility validation.

After OCR returns regions, predict the actual font size each region would
render at using font_replacer's real sizing function. If a region's predicted
size falls below a legibility threshold (12px body / 18px heading), split it
and re-check. Iterate until convergence or no more meaningful splits possible.

Heading classification: bbox height in the top 25% of all regions => heading.

The validator imports font_replacer.find_font_size as the single source of
truth — when sizing rules change in font_replacer.py, validator follows
automatically. No duplicated thresholds.
"""
import logging
from PIL import Image, ImageDraw

from font_replacer import find_font_size, get_font_path, BOLD_FONTS

logger = logging.getLogger(__name__)

BODY_MIN_PX = 12
HEADING_MIN_PX = 18
HEADING_PERCENTILE = 0.75  # top 25% of bbox heights = heading
MAX_ITERATIONS = 200


def validate_legibility(
    regions,
    font_name,
    body_min_px=BODY_MIN_PX,
    heading_min_px=HEADING_MIN_PX,
    heading_percentile=HEADING_PERCENTILE,
    max_iterations=MAX_ITERATIONS,
):
    """Iteratively split over-merged regions until every one renders legibly.

    Uses font_replacer.find_font_size as the source of truth for predicted
    font size — no duplicated logic, no drift.
    """
    if not regions:
        return regions

    font_path = get_font_path(font_name)
    is_bold = font_name in BOLD_FONTS

    # Heading cutoff: bbox heights at or above this percentile = heading.
    heights = sorted(r["bounding_box"]["height"] for r in regions)
    cutoff_idx = min(int(len(heights) * heading_percentile), len(heights) - 1)
    heading_cutoff = heights[cutoff_idx]

    # Dummy draw context for find_font_size (it only needs textbbox metrics).
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    output = []
    queue = list(regions)
    iterations = 0
    while queue and iterations < max_iterations:
        iterations += 1
        region = queue.pop(0)
        is_heading = region["bounding_box"]["height"] >= heading_cutoff
        min_px = heading_min_px if is_heading else body_min_px

        predicted = _predict_size(draw, region, font_path, is_bold)
        if predicted >= min_px:
            output.append(region)
            continue

        sub_regions = _split_region(region)
        if len(sub_regions) <= 1:
            logger.warning(
                "region_validator: region %r (bbox %dx%d, predicted %dpx, "
                "threshold %dpx) is unsplittable; accepting illegible.",
                region["text"][:40],
                region["bounding_box"]["width"],
                region["bounding_box"]["height"],
                predicted,
                min_px,
            )
            output.append(region)
        else:
            queue.extend(sub_regions)

    if queue:
        logger.warning(
            "region_validator: hit max_iterations=%d with %d regions still queued; "
            "accepting as-is.",
            max_iterations, len(queue),
        )
        output.extend(queue)

    return output


def _predict_size(draw, region, font_path, is_bold):
    """Predict what font size font_replacer would render this region at."""
    font = find_font_size(draw, region["text"], font_path, region["bounding_box"], is_bold)
    return font.size


def _split_region(region):
    """Split a region into two halves at a word boundary.

    Prefers `_words` metadata (Vision path) for accurate sub-bboxes; falls
    back to character-proportional bbox division when absent.
    """
    text = region["text"]
    bbox = region["bounding_box"]
    words = region.get("_words")

    if words and len(words) >= 2:
        mid = len(words) // 2
        left, right = words[:mid], words[mid:]
        return [
            _words_to_region(left, region),
            _words_to_region(right, region),
        ]

    # No word metadata: split text + bbox geometrically by char proportion.
    word_list = text.split()
    if len(word_list) < 2:
        return [region]
    mid = len(word_list) // 2
    left_text = " ".join(word_list[:mid])
    right_text = " ".join(word_list[mid:])
    total_chars = len(left_text) + len(right_text)
    if total_chars == 0:
        return [region]
    left_w = int(bbox["width"] * len(left_text) / total_chars)
    if left_w <= 0 or left_w >= bbox["width"]:
        return [region]
    confidence = region.get("confidence", 1.0)
    return [
        {
            "text": left_text,
            "confidence": confidence,
            "bounding_box": {
                "x": bbox["x"],
                "y": bbox["y"],
                "width": left_w,
                "height": bbox["height"],
            },
        },
        {
            "text": right_text,
            "confidence": confidence,
            "bounding_box": {
                "x": bbox["x"] + left_w,
                "y": bbox["y"],
                "width": bbox["width"] - left_w,
                "height": bbox["height"],
            },
        },
    ]


def _words_to_region(words, source_region):
    """Build a region dict from a list of word dicts (preserves _words)."""
    text = " ".join(w["text"] for w in words)
    x1 = min(w["x_min"] for w in words)
    x2 = max(w["x_max"] for w in words)
    y1 = min(w["y_min"] for w in words)
    y2 = max(w["y_max"] for w in words)
    return {
        "text": text,
        "confidence": source_region.get("confidence", 1.0),
        "bounding_box": {
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        },
        "_words": list(words),  # preserve for further splits if needed
    }
