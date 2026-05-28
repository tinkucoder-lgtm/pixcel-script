"""POST /api/generate-design — premium design generation via Gemini 3 Pro Image.

Validation rules (return 422 if violated):
  - description: required, 1–2000 chars
  - headline: required, 1–200 chars
  - subtext: optional, ≤300 chars
  - font_preset: optional; if present, must be one of the 8 valid presets
  - headline_font / body_font: optional; ≤100 chars each
  - cross-field: must provide font_preset OR both headline_font and body_font

Font resolution (after validation):
  - If font_preset provided → use its headline_font / body_font as defaults
  - If headline_font or body_font also provided → overrides the preset value
  - Custom-only mode (no preset) requires both headline_font and body_font

Errors from the underlying service are returned as JSON {error, detail} with
clean HTTP statuses — never a raw 500 or stack trace.
"""
import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from config.font_presets import FONT_PRESETS

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


@router.post("")
async def generate_design_endpoint(req: GenerateDesignRequest):
    # Font resolution: preset as base, custom overrides per-field
    headline_font = None
    body_font = None
    if req.font_preset:
        preset = FONT_PRESETS[req.font_preset]
        headline_font = preset["headline_font"]
        body_font = preset["body_font"]
    if req.headline_font:
        headline_font = req.headline_font
    if req.body_font:
        body_font = req.body_font
    # model_validator above guarantees both are now set

    try:
        from services.design_generator import generate_design, GenerationError
    except Exception as exc:
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
    except GenerationError as exc:
        logger.warning(
            "generate-design: %s (http %d) — %s", exc.kind, exc.http_status, exc.message
        )
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": exc.kind, "detail": exc.message},
        )
    except Exception as exc:
        # Defensive catch — the service should always raise GenerationError,
        # but if something slips through, surface it cleanly rather than a 500.
        logger.exception("generate-design: unexpected error escaped service")
        return JSONResponse(
            status_code=502,
            content={
                "error": "unexpected",
                "detail": f"{type(exc).__name__}: {str(exc)[:300]}",
            },
        )

    return {"image_url": result["output_url"]}
