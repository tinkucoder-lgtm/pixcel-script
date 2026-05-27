import os
from pathlib import Path
from statistics import median
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

CONFIDENCE_THRESHOLD = 0.4

_rapidocr_engine = None


def _setup_google_creds():
    key_path = Path(__file__).resolve().parent / "vision-key.json"
    if key_path.exists():
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(key_path))


def _normalize_conf(conf):
    try:
        value = float(conf)
        return value / 100.0 if value > 1 else value
    except Exception:
        return 0.0


def _build_region_from_tesseract(data, index):
    x = int(data["left"][index])
    y = int(data["top"][index])
    w = int(data["width"][index])
    h = int(data["height"][index])
    text = str(data["text"][index]).strip()
    confidence = _normalize_conf(data["conf"][index])
    return text, confidence, x, y, w, h


def _get_rapidocr_engine():
    """Lazy-init RapidOCR. PP-OCRv4 ONNX models are bundled in the
    rapidocr-onnxruntime wheel (no runtime model download). First call
    after server restart pays ~1-3s to load models into memory; subsequent
    calls reuse the cached engine instance.
    """
    global _rapidocr_engine
    if _rapidocr_engine is None:
        _rapidocr_engine = RapidOCR()
    return _rapidocr_engine


def _detect_with_rapidocr(image_path: str):
    """Detect text via RapidOCR (PP-OCRv4 served over ONNX runtime).

    Returns per-line regions with tight bboxes. RapidOCR's text detector
    naturally separates multi-column layouts at the line level, so the
    Vision-walk column splitter isn't needed here — each line/column is
    already its own detection.
    """
    engine = _get_rapidocr_engine()
    raw_results, _elapsed = engine(image_path)
    return _apply_second_pass(_rapidocr_results_to_regions(raw_results or []), image_path)


def _rapidocr_results_to_regions(raw_results):
    """Map RapidOCR raw output [[polygon, text, confidence], ...] to our
    region schema. Pure function — no I/O, no engine — so it can be unit
    tested without rapidocr-onnxruntime installed.

    Word-level bboxes aren't exposed by RapidOCR. We approximate them by
    distributing the line's bbox width proportionally to per-word character
    counts, populating `_words` so the post-OCR legibility validator has
    something to work with if it ever needs to split further.
    """
    regions = []
    for item in raw_results:
        bbox_polygon, text, confidence = item[0], item[1], item[2]
        confidence = float(confidence)
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        text = text.strip()
        if not text:
            continue

        xs = [p[0] for p in bbox_polygon]
        ys = [p[1] for p in bbox_polygon]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        width = int(round(x2 - x1))
        height = int(round(y2 - y1))
        if width <= 0 or height <= 0:
            continue

        words_meta = _build_word_metadata(text.split(), x1, y1, x2, y2)

        regions.append({
            "text": text,
            "confidence": round(confidence, 3),
            "bounding_box": {
                "x": int(round(x1)),
                "y": int(round(y1)),
                "width": width,
                "height": height,
            },
            "_words": words_meta,
        })

    return regions


def _build_word_metadata(line_words, x1, y1, x2, y2):
    """Approximate per-word bboxes by distributing line width proportionally
    to character counts. Used by both the pass-1 mapper and the pass-2
    translator so the schema stays consistent across paths."""
    if not line_words:
        return []
    char_lengths = [len(w) for w in line_words]
    total_chars = sum(char_lengths)
    if total_chars == 0:
        return []
    cursor = float(x1)
    line_width = float(x2 - x1)
    meta = []
    for w, n in zip(line_words, char_lengths):
        ww = line_width * n / total_chars
        meta.append({
            "text": w,
            "x_min": int(round(cursor)),
            "x_max": int(round(cursor + ww)),
            "y_min": int(round(y1)),
            "y_max": int(round(y2)),
        })
        cursor += ww
    return meta


def _text_looks_like_missing_spaces(text):
    """Heuristic: text is long enough to plausibly be a phrase, has no
    internal spaces, and mixes letters with a non-letter character class
    (digit, punctuation). The mixed-class condition lets us catch
    'NOTANMBA,ASTARTUPGARAGE' and 'MESAPROGRAMHIGHLIGHTS:' while skipping
    genuinely-single tokens like 'INTERNSHIPS' to keep the pass-2 call
    count low. False positives are cheap (pass-2 returns the original)."""
    if ' ' in text:
        return False
    if len(text) < 8:
        return False
    has_letter = any(c.isalpha() for c in text)
    has_other = any(not c.isalpha() and not c.isspace() for c in text)
    return has_letter and has_other


def _apply_second_pass(pass1_regions, image_path):
    """For each region whose text looks like missing spaces, crop and
    re-detect with tighter params. If pass-2 yields >=2 sub-detections
    within the crop, replace the original region with those sub-regions.
    Otherwise keep the original (pass-2 is a no-op for that region)."""
    if not any(_text_looks_like_missing_spaces(r["text"]) for r in pass1_regions):
        return pass1_regions
    engine = _get_rapidocr_engine()
    image = Image.open(image_path).convert("RGB")
    out = []
    for region in pass1_regions:
        if not _text_looks_like_missing_spaces(region["text"]):
            out.append(region)
            continue
        sub = _run_rapidocr_second_pass(image, region, engine)
        if sub is not None and len(sub) >= 2:
            out.extend(sub)
        else:
            out.append(region)
    return out


def _run_rapidocr_second_pass(image, region, engine):
    """Crop region's bbox out of the source image, re-run RapidOCR with
    tighter detection thresholds on the crop, translate any sub-bboxes back
    to image coordinates. Returns sub-region list or None if no usable split."""
    bbox = region["bounding_box"]
    pad = 4  # small margin so edge glyphs aren't clipped
    img_w, img_h = image.size
    crop_x = max(0, bbox["x"] - pad)
    crop_y = max(0, bbox["y"] - pad)
    crop_x2 = min(img_w, bbox["x"] + bbox["width"] + pad)
    crop_y2 = min(img_h, bbox["y"] + bbox["height"] + pad)
    crop = image.crop((crop_x, crop_y, crop_x2, crop_y2))
    import numpy as np
    crop_array = np.array(crop)
    # Tighter thresholds = more detections at finer granularity. Some RapidOCR
    # versions accept these kwargs at call-time; if not, the call falls back
    # to defaults (pass-2 becomes a no-op for that region rather than crash).
    try:
        raw, _ = engine(crop_array, box_thresh=0.3, unclip_ratio=1.3)
    except TypeError:
        raw, _ = engine(crop_array)
    if not raw or len(raw) < 2:
        return None
    return _translate_pass2_results(raw, region, crop_x, crop_y)


def _translate_pass2_results(raw_results, source_region, crop_x, crop_y):
    """Pure-function translator from pass-2 raw output (crop coords) to
    pass-1-schema sub-regions (image coords). Extracted for testability —
    no I/O, no engine — so tests can exercise pass-2 substitution with
    synthetic raw output."""
    sub_regions = []
    for item in raw_results:
        poly, text, conf = item[0], item[1], float(item[2])
        if conf < CONFIDENCE_THRESHOLD:
            continue
        text = text.strip()
        if not text:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1 = min(xs) + crop_x
        y1 = min(ys) + crop_y
        x2 = max(xs) + crop_x
        y2 = max(ys) + crop_y
        width = int(round(x2 - x1))
        height = int(round(y2 - y1))
        if width <= 0 or height <= 0:
            continue
        words_meta = _build_word_metadata(text.split(), x1, y1, x2, y2)
        sub_regions.append({
            "text": text,
            "confidence": round(conf, 3),
            "bounding_box": {
                "x": int(round(x1)),
                "y": int(round(y1)),
                "width": width,
                "height": height,
            },
            "_words": words_meta,
        })
    return sub_regions if len(sub_regions) >= 2 else None


def _detect_with_tesseract(image_path: str):
    image = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=Output.DICT, lang="eng")

    regions = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        text, confidence, x, y, w, h = _build_region_from_tesseract(data, i)
        if not text or confidence < CONFIDENCE_THRESHOLD:
            continue
        if w <= 0 or h <= 0:
            continue

        regions.append({
            "text": text,
            "confidence": round(confidence, 3),
            "bounding_box": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
        })

    return regions


def _group_words_into_lines(words):
    """Group word dicts into visual lines by y-coordinate overlap.

    Two words belong to the same line if their y-ranges overlap by at least
    half the smaller word's height. Within a line, words sort left-to-right
    by x_min; lines themselves sort top-to-bottom by y_min.
    """
    lines = []
    for word in sorted(words, key=lambda w: w["y_min"]):
        placed = False
        for line in lines:
            line_y_min = min(w["y_min"] for w in line)
            line_y_max = max(w["y_max"] for w in line)
            overlap = min(word["y_max"], line_y_max) - max(word["y_min"], line_y_min)
            min_h = min(word["y_max"] - word["y_min"], line_y_max - line_y_min)
            if min_h > 0 and overlap >= 0.5 * min_h:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])
    for line in lines:
        line.sort(key=lambda w: w["x_min"])
    lines.sort(key=lambda line: min(w["y_min"] for w in line))
    return lines


def _split_line_at_column_gaps(line, gap_multiplier=3.0, width_fraction=0.5):
    """Split a y-grouped line into column groups by detecting unusually large
    horizontal gaps between consecutive words.

    A gap qualifies as a column boundary if it exceeds:
        max(gap_multiplier * median_inter_word_gap,
            width_fraction * median_word_width)

    Using max() means the gap must look unusual under BOTH the relative-spacing
    signal AND the absolute-size signal, making the splitter conservative.

    Applied recursively to each sub-group so nested column-within-column
    structures get split too — a single line with multiple big gaps becomes
    3+ regions in one call.

    Known limitation: a 2-word line has only one gap, so its median equals
    that gap and the threshold becomes 3x that gap — meaning 2-word lines
    NEVER split. Intentional (no anomaly baseline with one data point);
    Mesa-class infographics have enough words per column that this is fine.
    """
    if len(line) < 2:
        return [line]
    # Words are pre-sorted by x_min (guaranteed by _group_words_into_lines).
    gaps = [line[i]["x_min"] - line[i - 1]["x_max"] for i in range(1, len(line))]
    widths = [w["x_max"] - w["x_min"] for w in line]
    median_gap = median(gaps)
    median_width = median(widths)
    threshold = max(gap_multiplier * median_gap, width_fraction * median_width)
    split_indices = [i + 1 for i, g in enumerate(gaps) if g > threshold]
    if not split_indices:
        return [line]
    groups = []
    start = 0
    for split in split_indices:
        groups.append(line[start:split])
        start = split
    groups.append(line[start:])
    # Recurse: each sub-group gets its own median/threshold. Catches
    # nested column structures where one outer gap masks medium-sized
    # inner column boundaries that only stand out in the sub-group.
    result = []
    for g in groups:
        result.extend(_split_line_at_column_gaps(g, gap_multiplier, width_fraction))
    return result


def _detect_with_google_vision(image_path: str):
    _setup_google_creds()
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Vision OCR failed: {response.error.message}")

    regions = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                # Collect every word in this paragraph with its bbox.
                words = []
                for word in paragraph.words:
                    text = "".join(s.text for s in word.symbols)
                    if not text:
                        continue
                    vertices = word.bounding_box.vertices
                    if not vertices:
                        continue
                    words.append({
                        "text": text,
                        "x_min": min(v.x for v in vertices),
                        "x_max": max(v.x for v in vertices),
                        "y_min": min(v.y for v in vertices),
                        "y_max": max(v.y for v in vertices),
                    })
                if not words:
                    continue
                # Two-stage grouping:
                #   1. y-overlap -> visual lines (fixes vertically-stacked
                #      role-titles that Vision merges into one paragraph).
                #   2. x-gap splitting within each line -> column groups
                #      (fixes horizontally-arranged timeline labels that
                #      share a y-range but belong to separate columns).
                for line in _group_words_into_lines(words):
                    for column in _split_line_at_column_gaps(line):
                        text = " ".join(w["text"] for w in column)
                        x1 = min(w["x_min"] for w in column)
                        x2 = max(w["x_max"] for w in column)
                        y1 = min(w["y_min"] for w in column)
                        y2 = max(w["y_max"] for w in column)
                        width = int(x2 - x1)
                        height = int(y2 - y1)
                        if width <= 0 or height <= 0:
                            continue
                        regions.append({
                            "text": text,
                            "confidence": 1.0,
                            "bounding_box": {
                                "x": int(x1),
                                "y": int(y1),
                                "width": width,
                                "height": height,
                            },
                            # Word-level metadata threaded through so the
                            # post-OCR legibility validator can split at
                            # real word boundaries (not char proportions).
                            "_words": list(column),
                        })

    return regions


def detect_text(image_path: str):
    # Preference order: Google Vision (if billing live) → RapidOCR (primary
    # local) → Tesseract (last resort). Each backend tried in turn; if it
    # raises, we record the error and fall through to the next.
    # A successful call returns immediately — no silent fall-through.
    errors = []

    if GOOGLE_VISION_AVAILABLE:
        try:
            return _detect_with_google_vision(image_path)
        except Exception as exc:
            errors.append(f"google_vision: {exc}")

    if RAPIDOCR_AVAILABLE:
        try:
            return _detect_with_rapidocr(image_path)
        except Exception as exc:
            errors.append(f"rapidocr: {exc}")

    if TESSERACT_AVAILABLE:
        try:
            return _detect_with_tesseract(image_path)
        except Exception as exc:
            errors.append(f"tesseract: {exc}")

    if errors:
        raise RuntimeError("All OCR backends failed: " + " | ".join(errors))
    raise RuntimeError(
        "No OCR backends are available. Install rapidocr-onnxruntime, "
        "pytesseract, or configure Google Vision."
    )
