"""Tests for /api/generate-design — validation, error mapping, classifiers.

Mocks the design-generation service so tests don't spend Vertex AI money.
Three layers covered:
  1. Pydantic validation rules (no service call needed; fail at 422)
  2. Router error-to-HTTP mapping (mock generate_design to raise
     GenerationError variants; verify clean JSON {error, detail} response)
  3. Pure classifier helpers (_classify_genai_error, _classify_finish_reason,
     build_prompt) — unit-tested directly without any mocking
"""
import asyncio
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from services.design_generator import (
    GenerationError,
    _classify_finish_reason,
    _classify_genai_error,
    build_prompt,
)
from main import app


client = TestClient(app)


def _valid_body():
    return {
        "description": "test description",
        "headline": "Test Headline",
        "font_preset": "editorial-elegant",
    }


# ---------------------------------------------------------------------------
# Validation tests (no service mocking needed — pydantic rejects pre-call)
# ---------------------------------------------------------------------------

def test_validation_missing_description():
    body = _valid_body()
    del body["description"]
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 422


def test_validation_empty_description():
    body = _valid_body()
    body["description"] = ""
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 422


def test_validation_description_too_long():
    body = _valid_body()
    body["description"] = "x" * 2001
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 422


def test_validation_headline_is_optional(monkeypatch):
    """headline used to be required; it's now optional. Validation should
    pass when omitted, and the service receives headline=None so the prompt
    asks Gemini to invent one."""
    captured: dict = {}

    async def fake_gen(**kwargs):
        captured.update(kwargs)
        return {"output_path": "outputs/fake.png",
                "output_url": "/outputs/fake.png",
                "reasoning_parts": []}
    monkeypatch.setattr("services.design_generator.generate_design", fake_gen)
    body = _valid_body()
    del body["headline"]
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 200
    # Service was called with headline=None (the absent field)
    assert captured.get("headline") is None


def test_validation_headline_too_long():
    body = _valid_body()
    body["headline"] = "x" * 201
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 422


def test_validation_subtext_too_long():
    body = _valid_body()
    body["subtext"] = "x" * 301
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 422


def test_validation_invalid_preset():
    body = _valid_body()
    body["font_preset"] = "nonexistent-preset"
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 422
    assert "Unknown font_preset" in r.text


def test_validation_no_font_source():
    body = _valid_body()
    del body["font_preset"]
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 422
    assert "Must provide" in r.text


def test_validation_headline_font_too_long():
    body = _valid_body()
    body["headline_font"] = "x" * 101
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 422


def test_validation_custom_fonts_only_passes_validation(monkeypatch):
    """No preset, but both custom fonts provided — should pass validation."""
    async def fake_gen(**kwargs):
        return {"output_path": "outputs/fake.png", "output_url": "/outputs/fake.png",
                "reasoning_parts": []}
    monkeypatch.setattr("services.design_generator.generate_design", fake_gen)
    body = {
        "description": "x",
        "headline": "y",
        "headline_font": "Helvetica",
        "body_font": "Times",
    }
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 200
    assert r.json() == {"image_url": "/outputs/fake.png"}


# ---------------------------------------------------------------------------
# Router error-mapping tests (mock generate_design; verify HTTP response)
# ---------------------------------------------------------------------------

def _patch_gen_raises(monkeypatch, error: GenerationError):
    async def fake(**kwargs):
        raise error
    monkeypatch.setattr("services.design_generator.generate_design", fake)


def test_route_maps_timeout_to_504(monkeypatch):
    _patch_gen_raises(monkeypatch, GenerationError("timeout", "exceeded 300s", 504))
    r = client.post("/api/generate-design", json=_valid_body())
    assert r.status_code == 504
    assert r.json() == {"error": "timeout", "detail": "exceeded 300s"}


def test_route_maps_rate_limit_to_429(monkeypatch):
    _patch_gen_raises(monkeypatch, GenerationError("rate_limit", "429 upstream", 429))
    r = client.post("/api/generate-design", json=_valid_body())
    assert r.status_code == 429
    assert r.json()["error"] == "rate_limit"


def test_route_maps_content_block_to_422(monkeypatch):
    _patch_gen_raises(monkeypatch, GenerationError("content_blocked", "SAFETY", 422))
    r = client.post("/api/generate-design", json=_valid_body())
    assert r.status_code == 422
    assert r.json()["error"] == "content_blocked"


def test_route_maps_no_image_to_502(monkeypatch):
    _patch_gen_raises(monkeypatch, GenerationError("no_image", "missing inline_data", 502))
    r = client.post("/api/generate-design", json=_valid_body())
    assert r.status_code == 502
    assert r.json()["error"] == "no_image"


def test_route_maps_unexpected_exception_to_502(monkeypatch):
    """Defensive: if service somehow raises a non-GenerationError, route
    still returns clean JSON rather than 500/stack trace."""
    async def fake_raises(**kwargs):
        raise ValueError("unexpected leak")
    monkeypatch.setattr("services.design_generator.generate_design", fake_raises)
    r = client.post("/api/generate-design", json=_valid_body())
    assert r.status_code == 502
    body = r.json()
    assert body["error"] == "unexpected"
    assert "ValueError" in body["detail"]


def test_route_happy_path_returns_image_url(monkeypatch):
    async def fake_gen(**kwargs):
        return {
            "output_path": "outputs/generated_abc.png",
            "output_url": "/outputs/generated_abc.png",
            "reasoning_parts": ["mock reasoning"],
        }
    monkeypatch.setattr("services.design_generator.generate_design", fake_gen)
    r = client.post("/api/generate-design", json=_valid_body())
    assert r.status_code == 200
    assert r.json() == {"image_url": "/outputs/generated_abc.png"}


# ---------------------------------------------------------------------------
# Pure classifier helpers (no mocks)
# ---------------------------------------------------------------------------

def test_classify_timeout_error_maps_to_504():
    err = _classify_genai_error(asyncio.TimeoutError(), timeout_s=300)
    assert err.kind == "timeout"
    assert err.http_status == 504
    assert "300" in err.message


def test_classify_unexpected_error_maps_to_502():
    err = _classify_genai_error(ValueError("oops"))
    assert err.kind == "unexpected"
    assert err.http_status == 502
    assert "ValueError" in err.message


def test_classify_finish_reason_stop_returns_none():
    from google.genai import types
    assert _classify_finish_reason(types.FinishReason.STOP) is None


def test_classify_finish_reason_safety_blocks_with_422():
    from google.genai import types
    err = _classify_finish_reason(types.FinishReason.SAFETY)
    assert err is not None
    assert err.kind == "content_blocked"
    assert err.http_status == 422


# ---------------------------------------------------------------------------
# Pure prompt builder (no mocks)
# ---------------------------------------------------------------------------

def test_build_prompt_includes_inputs():
    p = build_prompt(
        description="test desc",
        headline="My Headline",
        subtext=None,
        headline_font="serif",
        body_font="sans",
    )
    assert "test desc" in p
    assert "My Headline" in p
    assert "serif" in p
    assert "sans" in p


def test_build_prompt_subtext_included_when_provided():
    p = build_prompt("d", "h", "my subtext text", "s", "s")
    assert "my subtext text" in p


def test_build_prompt_subtext_omitted_when_none():
    p = build_prompt("d", "h", None, "s", "s")
    assert "supporting text verbatim" not in p


def test_build_prompt_appends_anti_ai_block():
    from config.font_presets import ANTI_AI_DESIGN
    p = build_prompt("d", "h", None, "s", "s")
    assert ANTI_AI_DESIGN in p


def test_build_prompt_falls_back_when_headline_is_none():
    """When headline is omitted, prompt instructs Gemini to invent one
    rather than rendering a design with no title."""
    p = build_prompt("d", None, "subtext", "s", "b")
    assert "Create an appropriate, attention-grabbing headline" in p
    # And does NOT include the verbatim-headline directive
    assert "must prominently feature this exact headline" not in p


def test_build_prompt_falls_back_when_subtext_is_none():
    """When subtext is omitted, prompt asks Gemini to add fitting supporting
    text rather than leaving the design caption-less."""
    p = build_prompt("d", "h", None, "s", "b")
    assert "Add fitting supporting text" in p


def test_build_prompt_includes_equally_important_body_bullet():
    """The typography rules block must call out body style as equally
    important as headline — otherwise Gemini deprioritizes body fonts."""
    p = build_prompt("d", "h", None, "s", "b")
    assert "EQUALLY IMPORTANT" in p


def test_previous_image_url_with_invalid_prefix_returns_400(monkeypatch):
    """Reject anything that doesn't start with /outputs/ — path-traversal
    guard."""
    async def fake_gen(**kwargs):  # service should never be called
        raise AssertionError("service must not be called for invalid input")
    monkeypatch.setattr("services.design_generator.generate_design", fake_gen)
    body = _valid_body()
    body["previous_image_url"] = "/etc/passwd"
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_previous_image_url"


def test_previous_image_url_with_path_traversal_returns_400(monkeypatch):
    async def fake_gen(**kwargs):
        raise AssertionError("service must not be called for invalid input")
    monkeypatch.setattr("services.design_generator.generate_design", fake_gen)
    body = _valid_body()
    body["previous_image_url"] = "/outputs/../etc/passwd"
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 400


def test_previous_image_url_nonexistent_returns_404(monkeypatch):
    async def fake_gen(**kwargs):
        raise AssertionError("service must not be called when file missing")
    monkeypatch.setattr("services.design_generator.generate_design", fake_gen)
    body = _valid_body()
    body["previous_image_url"] = "/outputs/this-file-definitely-does-not-exist.png"
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 404
    assert r.json()["error"] == "previous_image_not_found"


def test_previous_image_url_valid_passes_bytes_to_service(monkeypatch, tmp_path):
    """Valid /outputs/<filename> reads the file and forwards bytes to the
    service via previous_image_bytes."""
    img = tmp_path / "fixture.png"
    fake_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes-for-test"
    img.write_bytes(fake_bytes)
    # Point the router's outputs_dir lookup at tmp_path
    monkeypatch.setattr("config.settings.settings.outputs_dir", str(tmp_path))

    captured: dict = {}
    async def fake_gen(**kwargs):
        captured.update(kwargs)
        return {"output_path": "outputs/fake.png", "output_url": "/outputs/fake.png",
                "reasoning_parts": []}
    monkeypatch.setattr("services.design_generator.generate_design", fake_gen)

    body = _valid_body()
    body["previous_image_url"] = "/outputs/fixture.png"
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 200
    assert captured.get("previous_image_bytes") == fake_bytes
    assert captured.get("previous_image_mime") == "image/png"


def test_omitted_previous_image_url_means_no_bytes_to_service(monkeypatch):
    """Default behavior: no previous_image_url → previous_image_bytes is
    None when the service is called."""
    captured: dict = {}
    async def fake_gen(**kwargs):
        captured.update(kwargs)
        return {"output_path": "outputs/fake.png", "output_url": "/outputs/fake.png",
                "reasoning_parts": []}
    monkeypatch.setattr("services.design_generator.generate_design", fake_gen)
    body = _valid_body()
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 200
    assert captured.get("previous_image_bytes") is None


def test_compress_reference_image_shrinks_large_input():
    """Refinement reference must be small enough that sending it to Gemini
    doesn't blow up latency. Helper must produce something meaningfully
    smaller than the input and clamp the longest side to 512px."""
    from PIL import Image
    import io
    from services.design_generator import _compress_reference_image

    # Synthesize a large PNG (2K square with structure so JPEG can't trivially
    # collapse it — solid colors compress to near-zero and don't represent
    # real images).
    img = Image.new("RGB", (2048, 2048))
    pixels = img.load()
    for y in range(0, 2048, 32):
        for x in range(0, 2048, 32):
            pixels[x, y] = ((x * 31) % 255, (y * 17) % 255, ((x + y) * 13) % 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    big_bytes = buf.getvalue()

    compressed, mime = _compress_reference_image(big_bytes)

    assert mime == "image/jpeg"
    # At minimum 5x smaller for any non-trivial image
    assert len(compressed) < len(big_bytes) / 5, \
        f"compressed too large: {len(compressed)} vs {len(big_bytes)}"
    # Output must be a valid JPEG with longest side <= 512
    out = Image.open(io.BytesIO(compressed))
    assert out.format == "JPEG"
    assert max(out.size) <= 512


def test_compress_reference_image_handles_rgba_input():
    """JPEG can't carry transparency — helper must flatten RGBA to RGB
    before save() or PIL raises OSError."""
    from PIL import Image
    import io
    from services.design_generator import _compress_reference_image

    img = Image.new("RGBA", (1024, 1024), (255, 100, 50, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rgba_bytes = buf.getvalue()

    # Must not raise OSError("cannot write mode RGBA as JPEG")
    compressed, mime = _compress_reference_image(rgba_bytes)
    assert mime == "image/jpeg"
    assert len(compressed) > 0


def test_modification_prelude_includes_required_text():
    """When refining, the prelude must EXPLICITLY tell Gemini it's modifying
    an existing image, not generating fresh. Otherwise it ignores the input
    image and generates something completely different."""
    from services.design_generator import MODIFICATION_PRELUDE
    assert "MODIFYING an existing design" in MODIFICATION_PRELUDE
    assert "Keep the same layout" in MODIFICATION_PRELUDE
    assert "ONLY change what the user specifically asked" in MODIFICATION_PRELUDE
    assert "Do NOT create a completely new design" in MODIFICATION_PRELUDE


def test_generate_design_does_not_crash_on_none_headline(monkeypatch):
    """Regression for the 502 caused by `headline[:60]` in a log line.

    Mocks _get_client to raise a marker exception AFTER the log line, so we
    never actually hit Gemini — but the log line itself MUST run cleanly
    first. If headline=None crashes that line again, this test fails with
    TypeError instead of the marker.
    """
    import asyncio as _asyncio

    class _MarkerError(Exception):
        pass

    def fake_client():
        raise _MarkerError("marker — proves log line ran")

    monkeypatch.setattr("services.design_generator._get_client", fake_client)

    from services.design_generator import generate_design as gen

    with pytest.raises(_MarkerError):
        _asyncio.run(gen(
            description="test",
            headline=None,
            subtext=None,
            headline_font="serif",
            body_font="sans",
        ))


def test_anti_ai_block_includes_full_bleed_directive():
    """Step-1 change: full-bleed instruction must be in ANTI_AI_DESIGN."""
    from config.font_presets import ANTI_AI_DESIGN
    assert "full-bleed" in ANTI_AI_DESIGN.lower()
    assert "mockup" in ANTI_AI_DESIGN.lower()
