"""Phase 0 smoke test: Gemini 3 Pro Image via Vertex AI.

Standalone — does NOT touch any existing pipeline code. Verifies that:
  STEP A — auth works (cheap text-only call to gemini-2.5-flash)
  STEP B — image generation works (gemini-3-pro-image-preview)

Run:  cd ~/pixelscript/backend && venv/bin/python tests/gemini_smoke_test.py
"""
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent

# Use existing vision-key.json for auth (same service account that works for Vision API).
KEY_PATH = BACKEND / "vision-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)
print(f"GOOGLE_APPLICATION_CREDENTIALS = {KEY_PATH}")
print(f"Key file exists: {KEY_PATH.exists()}")
print()

from google import genai
from google.genai import types

# Per notebook example: vertexai=True (= enterprise=True alias), project + global location.
client = genai.Client(vertexai=True, project="pixelscript-prod", location="global")
print("Client initialized: vertexai=True, project=pixelscript-prod, location=global")
print()

# ---------------------------------------------------------------------------
# STEP A — cheap text-only call (verifies AUTH before image gen, which is $$)
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP A: text-only call to gemini-2.5-flash")
print("=" * 60)
try:
    resp_a = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="reply with OK",
    )
    print(f"  text response: {resp_a.text!r}")
    print(f"  STEP A SUCCESS")
except Exception as e:
    print(f"  STEP A FAILED: {type(e).__name__}")
    print(f"  full error: {e}")
    traceback.print_exc()
    sys.exit(1)
print()

# ---------------------------------------------------------------------------
# STEP B — image generation via gemini-3-pro-image-preview
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP B: image gen via gemini-3-pro-image-preview")
print("=" * 60)
output_path = HERE / "gemini_smoke_output.png"
try:
    resp_b = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents="a single coffee cup on a white background",
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(aspect_ratio="1:1"),
        ),
    )

    if not resp_b.candidates:
        print(f"  STEP B FAILED: no candidates in response")
        sys.exit(2)

    cand = resp_b.candidates[0]
    print(f"  finish_reason: {cand.finish_reason}")
    if cand.finish_reason != types.FinishReason.STOP:
        print(f"  WARNING: non-STOP finish reason — image may be partial/missing")

    image_bytes = None
    text_parts = []
    for part in (cand.content.parts or []):
        if getattr(part, "text", None):
            text_parts.append(part.text)
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            image_bytes = inline.data
            print(f"  inline_data mime_type: {getattr(inline, 'mime_type', '?')}")

    if text_parts:
        print(f"  text parts: {' | '.join(t[:100] for t in text_parts)}")

    if image_bytes:
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        size = output_path.stat().st_size
        print(f"  image saved: {output_path}")
        print(f"  image size: {size} bytes")
        print(f"  STEP B SUCCESS")
    else:
        print(f"  STEP B FAILED: response had no inline_data with image bytes")
        sys.exit(3)
except Exception as e:
    print(f"  STEP B FAILED: {type(e).__name__}")
    print(f"  full error: {e}")
    traceback.print_exc()
    sys.exit(4)
