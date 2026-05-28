"""POST /api/generate-design — premium design generation via Gemini 3 Pro Image.

Validation rules (return 422 if violated):
  - description: required, 1–2000 chars
  - headline: required, 1–200 chars
  - subtext: optional, ≤300 chars
  - font_preset: optional; if present, must be one of the 8 valid presets
  - headline_font / body_font: optional; ≤100 chars each
  - cross-field: must provide font_preset OR both headline_font and body_font

Rate limiting: per-IP via slowapi using settings.rate_limit_str.
Excess requests return 429 with {"error": "rate_limit_exceeded", "detail": ...}.

Structured logging: one JSON line per request (success or error) for usage +
spend monitoring. See `_emit_generation_event` for schema.
"""
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from config.font_presets import FONT_PRESETS
from config.limiter import limiter
from config.settings import settings

router = APIRouter(prefix="/api/generate-design", tags=["generate-design"])
logger = logging.getLogger(__name__)

VALID_PRESETS = frozenset(FONT_PRESETS.keys())


class GenerateDesignRequest(BaseModel):
    description: str = Field(min_length=1, max_length=2000)
    headline: str = Field(min_length=1, max_length=200)
    subtext: Optional[str] = Field(default=None, max_length=300)
    font_preset: Optional[str] = None
    headline_font: Optional[str] = Field(default=None, max_length=100)
    body_font: Optional[str] = Field(default=None, max_length=100)

    @field_validator("font_preset")
    @classmethod
    def validate_preset_name(cls, v):
        if v is not None and v not in VALID_PRESETS:
            raise ValueError(
                f"Unknown font_preset {v!r}. Valid: {sorted(VALID_PRESETS)}"
            )
        return v

    @model_validator(mode="after")
    def validate_font_source(self):
        if not self.font_preset and not (self.headline_font and self.body_font):
            raise ValueError(
                "Must provide font_preset, or both headline_font and body_font directly."
            )
        return self


def _emit_generation_event(
    *,
    description: str,
    preset: str,
    latency_s: float,
    outcome: str,
    cost_usd: float,
) -> None:
    """Emit one structured JSON line per generation event for monitoring."""
    event = {
        "event": "generation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": description[:100],
        "preset": preset,
        "latency_s": round(latency_s, 2),
        "outcome": outcome,
        "estimated_cost_usd": round(cost_usd, 4),
    }
    logger.info("generation_event %s", json.dumps(event, ensure_ascii=False))


@router.post("")
@limiter.limit(settings.rate_limit_str)
async def generate_design_endpoint(request: Request, req: GenerateDesignRequest):
    # Resolve fonts (preset as base, custom overrides per-field)
    headline_font = body_font = None
    if req.font_preset:
        preset_def = FONT_PRESETS[req.font_preset]
        headline_font = preset_def["headline_font"]
        body_font = preset_def["body_font"]
    if req.headline_font:
        headline_font = req.headline_font
    if req.body_font:
        body_font = req.body_font

    preset_label = req.font_preset or "custom"
    t0 = time.perf_counter()

    try:
        from services.design_generator import generate_design, GenerationError
    except Exception as exc:
        latency = time.perf_counter() - t0
        _emit_generation_event(
            description=req.description, preset=preset_label,
            latency_s=latency, outcome="import_failed", cost_usd=0.0,
        )
        logger.exception("generate-design: dependency import failed")
        return JSONResponse(
            status_code=500,
            content={"error": "import_failed", "detail": str(exc)[:300]},
        )

    try:
        result = await generate_design(
            description=req.description,
            headline=req.headline,
            subtext=req.subtext,
            headline_font=headline_font,
            body_font=body_font,
        )
        latency = time.perf_counter() - t0
        _emit_generation_event(
            description=req.description, preset=preset_label,
            latency_s=latency, outcome="success",
            cost_usd=settings.estimated_cost_per_image_usd,
        )
        return {"image_url": result["output_url"]}
    except GenerationError as exc:
        latency = time.perf_counter() - t0
        _emit_generation_event(
            description=req.description, preset=preset_label,
            latency_s=latency, outcome=exc.kind, cost_usd=0.0,
        )
        logger.warning(
            "generate-design: %s (http %d) — %s", exc.kind, exc.http_status, exc.message
        )
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": exc.kind, "detail": exc.message},
        )
    except Exception as exc:
        latency = time.perf_counter() - t0
        _emit_generation_event(
            description=req.description, preset=preset_label,
            latency_s=latency, outcome="unexpected", cost_usd=0.0,
        )
        logger.exception("generate-design: unexpected error escaped service")
        return JSONResponse(
            status_code=502,
            content={
                "error": "unexpected",
                "detail": f"{type(exc).__name__}: {str(exc)[:300]}",
            },
        )
