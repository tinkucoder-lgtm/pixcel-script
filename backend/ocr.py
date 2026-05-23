import os
from pathlib import Path
from PIL import Image

try:
    import easyocr
    import numpy as np
    EASY_OCR_AVAILABLE = True
except ImportError:
    EASY_OCR_AVAILABLE = False

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

reader = None
CONFIDENCE_THRESHOLD = 0.4


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


def _get_easyocr_reader():
    global reader
    if "reader" not in globals() or globals().get("reader") is None:
        globals()["reader"] = easyocr.Reader(["en"], gpu=False)
    return globals()["reader"]


def _detect_with_easyocr(image_path: str):
    reader = _get_easyocr_reader()
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    raw_results = reader.readtext(image_array, detail=1, paragraph=False)

    regions = []
    for bbox, text, confidence in raw_results:
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        bbox = np.array(bbox, dtype=float)
        x1, y1 = bbox[:, 0].min(), bbox[:, 1].min()
        x2, y2 = bbox[:, 0].max(), bbox[:, 1].max()
        width = int(round(max(0.0, x2 - x1)))
        height = int(round(max(0.0, y2 - y1)))

        if width <= 0 or height <= 0:
            continue

        regions.append({
            "text": text.strip(),
            "confidence": round(float(confidence), 3),
            "bounding_box": {
                "x": int(round(x1)),
                "y": int(round(y1)),
                "width": width,
                "height": height,
            }
        })

    return regions


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
            text = "".join(
                symbol.text
                for paragraph in block.paragraphs
                for word in paragraph.words
                for symbol in word.symbols
            ).strip()
            if not text:
                continue

            vertices = block.bounding_box.vertices
            x1 = min(v.x for v in vertices)
            y1 = min(v.y for v in vertices)
            x2 = max(v.x for v in vertices)
            y2 = max(v.y for v in vertices)
            width = int(x2 - x1)
            height = int(y2 - y1)
            if width <= 0 or height <= 0:
                continue

            regions.append({
                "text": text,
                "confidence": 1.0,
                "bounding_box": {
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                }
            })

    return regions


def detect_text(image_path: str):
    if EASY_OCR_AVAILABLE:
        return _detect_with_easyocr(image_path)

    if TESSERACT_AVAILABLE:
        return _detect_with_tesseract(image_path)

    if GOOGLE_VISION_AVAILABLE:
        return _detect_with_google_vision(image_path)

    raise RuntimeError("No OCR backends are available. Install easyocr, pytesseract, or configure Google Vision.")
