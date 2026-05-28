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
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

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
    # headline and subtext are both optional. The frontend now omits them
    # entirely; legacy callers that still send them get honored. When omitted,
    # the service instructs Gemini to invent appropriate text rather than
    # rendering a blank slate.
    headline: Optional[str] = Field(default=None, max_length=200)
    subtext: Optional[str] = Field(default=None, max_length=300)
    font_preset: Optional[str] = None
    headline_font: Optional[str] = Field(default=None, max_length=100)
    body_font: Optional[str] = Field(default=None, max_length=100)
    # Refinement: when supplied, the named file is loaded and sent to Gemini
    # alongside the prompt so it modifies the existing design rather than
    # creating a new one. Must be a /outputs/<filename> path we already serve;
    # path traversal is rejected.
    previous_image_url: Optional[str] = Field(default=None, max_length=500)

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


def _load_previous_image(url: Optional[str]) -> Tuple[Optional[bytes], Optional[str], Optional[JSONResponse]]:
    """Validate previous_image_url + load bytes off disk safely.

    Returns (bytes, mime, None) on success or (None, None, JSONResponse) on
    error (so the caller can early-return the response). Path traversal is
    rejected: only bare filenames under /outputs/ are accepted.
    """
    if not url:
        return None, None, None
    url = url.strip()
    prefix = "/outputs/"
    if not url.startswith(prefix):
        return None, None, JSONResponse(
            status_code=400,
            content={
                "error": "invalid_previous_image_url",
                "detail": f"previous_image_url must start with {prefix!r}",
            },
        )
    filename = url[len(prefix):]
    # Reject path traversal: no separators, no .., no empty
    if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
        return None, None, JSONResponse(
            status_code=400,
            content={
                "error": "invalid_previous_image_url",
                "detail": "previous_image_url filename is invalid",
            },
        )
    full_path = Path(settings.outputs_dir) / filename
    if not full_path.is_file():
        return None, None, JSONResponse(
            status_code=404,
            content={
                "error": "previous_image_not_found",
                "detail": f"Previous image file not found: {filename}",
            },
        )
    try:
        data = full_path.read_bytes()
    except Exception as exc:
        return None, None, JSONResponse(
            status_code=500,
            content={
                "error": "previous_image_read_failed",
                "detail": str(exc)[:200],
            },
        )
    mime, _ = mimetypes.guess_type(str(full_path))
    return data, (mime or "image/png"), None


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

    # Load previous-image bytes if the caller is refining
    previous_image_bytes, previous_image_mime, prev_err_response = _load_previous_image(req.previous_image_url)
    if prev_err_response is not None:
        return prev_err_response

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
            previous_image_bytes=previous_image_bytes,
            previous_image_mime=previous_image_mime or "image/png",
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
