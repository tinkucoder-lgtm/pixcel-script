"""Phase 1, step 1 — image-EDITING smoke test for Gemini 3 Pro Image.

Standalone — does NOT touch any existing pipeline code. Verifies that the
image-to-image (editing) call pattern works, using the same client init that
Phase 0's text+image smoke test confirmed.

Pattern is from the official notebook (intro_gemini_3_image_gen.ipynb,
cell 29 — image editing section): pass the source image as a Part.from_bytes
in `contents`, followed by the editing instruction string.

Run:  cd ~/pixelscript/backend && venv/bin/python tests/gemini_edit_test.py
"""
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent

# Same auth setup as Phase 0
KEY_PATH = BACKEND / "vision-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)
print(f"GOOGLE_APPLICATION_CREDENTIALS = {KEY_PATH}")

SOURCE_PATH = Path.home() / "Downloads" / "IMG_4226.JPG"
OUTPUT_PATH = HERE / "gemini_edit_output2.png"
MODEL_ID = "gemini-3-pro-image-preview"

EDIT_PROMPT = (
    "Redesign the TYPOGRAPHY of this poster comprehensively. You MUST "
    "change the font of every text element to a cohesive new typeface "
    "family: the headline 'WAFFLES ARE JUST', the large word 'PANCAKES', "
    "the 'WITH ABS' label, the 'YaMito' wordmark, and both 'Where every "
    "bite drips happiness' taglines. Use an elegant, refined editorial "
    "serif typeface throughout — the kind seen on a premium food magazine "
    "cover like Bon Appétit or Kinfolk — replacing the current bold/brush "
    "fonts entirely. CRITICAL: keep the waffle photograph, the layout, "
    "the colors, the gold logo emblem, the paint-stroke banners, the "
    "decorative frames and icons, and the overall composition exactly as "
    "they are. Change ONLY the fonts of the text, but change ALL of them. "
    "The result should look like the same poster redesigned by a high-end "
    "typographer."
)

print(f"Source image: {SOURCE_PATH}  (exists: {SOURCE_PATH.exists()})")
print(f"Output target: {OUTPUT_PATH}")
print(f"Model: {MODEL_ID}")
print()

if not SOURCE_PATH.exists():
    print(f"FATAL: source image not found at {SOURCE_PATH}")
    sys.exit(1)

from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="pixelscript-prod", location="global")
print("Client initialized: vertexai=True, project=pixelscript-prod, location=global")
print()

# Load source as bytes (notebook pattern)
with open(SOURCE_PATH, "rb") as f:
    source_bytes = f.read()
src_size = len(source_bytes)
mime_type = "image/jpeg"  # source is .JPG
print(f"Source loaded: {src_size} bytes, mime={mime_type}")
print()

# ---------------------------------------------------------------------------
# The image-edit call
# ---------------------------------------------------------------------------
print("=" * 60)
print("Sending edit request...")
print("=" * 60)
try:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=source_bytes, mime_type=mime_type),
            EDIT_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(image_size="2K"),
        ),
    )
except Exception as e:
    print(f"REQUEST FAILED: {type(e).__name__}")
    print(f"full error: {e}")
    traceback.print_exc()
    sys.exit(2)

if not response.candidates:
    print("FAILED: no candidates returned")
    sys.exit(3)

cand = response.candidates[0]
print(f"finish_reason: {cand.finish_reason}")
if cand.finish_reason != types.FinishReason.STOP:
    print(f"WARNING: non-STOP finish reason — output may be partial")
print()

# Separate inline_data (image) from text (reasoning) per Phase 0 finding
image_bytes = None
inline_mime = None
text_parts = []
for part in (cand.content.parts or []):
    if getattr(part, "text", None):
        text_parts.append(part.text)
    inline = getattr(part, "inline_data", None)
    if inline and getattr(inline, "data", None):
        image_bytes = inline.data
        inline_mime = getattr(inline, "mime_type", None)

print("=" * 60)
print(f"REASONING PARTS ({len(text_parts)} text parts returned)")
print("=" * 60)
for i, t in enumerate(text_parts):
    print(f"--- text part {i+1} ---")
    print(t)
    print()

print("=" * 60)
print("IMAGE PART")
print("=" * 60)
if not image_bytes:
    print("FAILED: no image bytes in response")
    sys.exit(4)
print(f"inline_data mime_type: {inline_mime}")
with open(OUTPUT_PATH, "wb") as f:
    f.write(image_bytes)
out_size = OUTPUT_PATH.stat().st_size
print(f"Saved to: {OUTPUT_PATH}")
print(f"Output size: {out_size} bytes")
print(f"Size ratio (output/source): {out_size/src_size:.2f}x")
print()
print("SUCCESS")
