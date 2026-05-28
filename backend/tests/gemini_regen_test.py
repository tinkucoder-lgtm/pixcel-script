"""Phase 1, step 3 — full REGENERATION test with source as reference.

Distinct from gemini_edit_test.py which constrains the model to edit-in-place.
Here we hand the model the source image as a brand/content reference and grant
full creative freedom on layout, fonts, and composition — only the brand,
message, and waffle subject must remain.

Same client init + auth + image-as-Part pattern as gemini_edit_test.py.

Run:  cd ~/pixelscript/backend && venv/bin/python tests/gemini_regen_test.py
"""
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent

KEY_PATH = BACKEND / "vision-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)
print(f"GOOGLE_APPLICATION_CREDENTIALS = {KEY_PATH}")

# User specified ~/Downloads/yamito_source.jpg; the on-disk file is IMG_4226.JPG.
SOURCE_PATH = Path.home() / "Downloads" / "IMG_4226.JPG"
OUTPUT_PATH = HERE / "gemini_regen_output.png"
MODEL_ID = "gemini-3-pro-image-preview"

REGEN_PROMPT = (
    "Using this poster as a reference for the brand and content, create a "
    "NEW, premium version of this advertisement. Keep the same brand "
    "(YaMito Bites), the same core message ('Waffles are just pancakes "
    "with abs', 'Where every bite drips happiness'), the same fitness-"
    "meets-breakfast theme, and a stack of waffles with berries as the "
    "hero subject. But redesign it completely with sophisticated, "
    "intentional, professional typography and elevated art direction — "
    "the kind a top-tier creative agency would produce for a premium "
    "brand. It should look unmistakably human-designed and high-end, not "
    "generic AI output. You have full creative freedom on fonts, layout, "
    "and composition as long as the brand, message, and waffle subject "
    "remain."
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

with open(SOURCE_PATH, "rb") as f:
    source_bytes = f.read()
src_size = len(source_bytes)
mime_type = "image/jpeg"
print(f"Source loaded: {src_size} bytes, mime={mime_type}")
print()

print("=" * 60)
print("Sending regeneration request...")
print("=" * 60)
try:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=source_bytes, mime_type=mime_type),
            REGEN_PROMPT,
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
