"""Phase 1a — prove the core generation loop before building any endpoint.

Standalone — does NOT touch any existing pipeline code. Same client init as
gemini_smoke_test.py (vertexai=True, project=pixelscript-prod, location=global,
model gemini-3-pro-image-preview).

This is text-to-image (no source image). Prompt is split into two parts:
  - INPUT BLOCK: the three values (description, headline, font_style) that
    would later become parameters of a real endpoint
  - ANTI-AI-DESIGN BLOCK: fixed text appended to every call; encodes what
    makes generic AI-generated design look generic, and explicitly forbids it

Once this loop works visibly, the same `build_prompt` function gets reused
verbatim in the eventual endpoint with the three values coming from the API
request body.

Run: cd ~/pixelscript/backend && venv/bin/python tests/gemini_generate_test.py
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

OUTPUT_PATH = HERE / "gemini_generate_output.png"
MODEL_ID = "gemini-3-pro-image-preview"

# Hardcoded sample inputs (would be endpoint parameters later)
INPUT_DESCRIPTION = (
    "Instagram post for an artisan coffee roaster announcing a new "
    "single-origin Ethiopian roast"
)
INPUT_HEADLINE = "Ethiopian Sunrise — Now Roasting"
INPUT_FONT_STYLE = "elegant high-contrast editorial serif"


def build_prompt(description: str, headline: str, font_style: str) -> str:
    """Compose the full generation prompt. Per-call inputs first, then the
    fixed anti-AI-design block that should never change between calls."""
    return (
        f"Create a professional, premium Instagram post design for: "
        f"{description}. "
        f"The design must prominently feature this exact headline, spelled "
        f"exactly: '{headline}'. "
        f"Typography: use {font_style} for the headline, plus at most one "
        f"complementary supporting font. "
        f"ART DIRECTION — this must look like the work of a senior human "
        f"designer, NOT AI-generated. Follow these rules strictly: "
        f"intentional asymmetric composition with a single clear focal "
        f"point (never centered symmetry); restrained purposeful decoration "
        f"with NO gratuitous glows, sparkles, or filler ornaments; a "
        f"cohesive type system, not a grab-bag of fonts; generous "
        f"intentional whitespace and clear visual hierarchy; sophisticated "
        f"restrained color, avoid oversaturation and heavy vignettes; clean "
        f"real-looking photography or intentional illustration, never "
        f"AI-merged hybrid objects; overall premium editorial deliberately-"
        f"designed feel. Render ALL text crisply, legibly, and with correct "
        f"spelling."
    )


prompt = build_prompt(INPUT_DESCRIPTION, INPUT_HEADLINE, INPUT_FONT_STYLE)
print(f"Output target: {OUTPUT_PATH}")
print(f"Model: {MODEL_ID}")
print(f"Prompt length: {len(prompt)} chars")
print()
print("=" * 60)
print("FULL PROMPT (verbatim — what the model receives)")
print("=" * 60)
print(prompt)
print()

from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="pixelscript-prod", location="global")
print("Client initialized: vertexai=True, project=pixelscript-prod, location=global")
print()

print("=" * 60)
print("Sending generation request...")
print("=" * 60)
try:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(
                aspect_ratio="1:1",  # Instagram post
                image_size="2K",
            ),
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
print()
print("SUCCESS")
