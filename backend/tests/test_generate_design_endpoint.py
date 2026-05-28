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


def test_validation_missing_headline():
    body = _valid_body()
    del body["headline"]
    r = client.post("/api/generate-design", json=body)
    assert r.status_code == 422


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


def test_anti_ai_block_includes_full_bleed_directive():
    """Step-1 change: full-bleed instruction must be in ANTI_AI_DESIGN."""
    from config.font_presets import ANTI_AI_DESIGN
    assert "full-bleed" in ANTI_AI_DESIGN.lower()
    assert "mockup" in ANTI_AI_DESIGN.lower()
