"""Design-generation service — wraps Gemini 3 Pro Image via Vertex AI (async).

Async path: uses the genai SDK's `.aio.models.generate_content` so concurrent
requests don't serialize on uvicorn's single worker. A 5-min timeout via
asyncio.wait_for kills truly-hung Vertex calls (normal range is 30–90s; the
once-seen 9.5-min outlier suggests bounded-but-occasionally-long is real).

Public surface:
  - build_prompt(...) — pure function; testable without the API
  - generate_design(...) — async; raises GenerationError on any failure
  - GenerationError — service-level exception carrying kind/message/http_status
  - _classify_genai_error / _classify_finish_reason — pure classifier helpers
    extracted so error mapping is unit-testable

Heavy SDK imports (google.genai) are deferred to the call site so module
import stays cheap for any code path that doesn't actually generate.
"""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from config.font_presets import ANTI_AI_DESIGN
from config.settings import settings
from services.storage import LocalStorage

logger = logging.getLogger(__name__)

# Module-level singletons; lazy-init the heavy ones.
_client = None
_storage = LocalStorage(settings.outputs_dir)


class GenerationError(Exception):
    """Service-level error. Carries http_status so the router can map cleanly
    without re-inspecting the underlying SDK exception type."""

    def __init__(self, kind: str, message: str, http_status: int):
        self.kind = kind
        self.message = message
        self.http_status = http_status
        super().__init__(f"{kind}: {message}")


def _setup_google_creds():
    """Idempotent — same pattern as ocr.py. setdefault means an externally-set
    credential path takes precedence."""
    import os
    key_path = Path(__file__).resolve().parent.parent / "vision-key.json"
    if key_path.exists():
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(key_path))


def _get_client():
    """Lazy-init singleton Vertex AI client (sync object; async ops via .aio)."""
    global _client
    if _client is None:
        _setup_google_creds()
        from google import genai
        _client = genai.Client(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.vertex_location,
        )
    return _client


def build_prompt(
    description: str,
    headline: str,
    subtext: Optional[str],
    headline_font: str,
    body_font: str,
) -> str:
    """Compose the full generation prompt: per-call inputs first, then the
    fixed ANTI_AI_DESIGN block. Pure function — no I/O, no API calls."""
    parts = [
        f"Create a professional, premium design for: {description}.",
        f"The design must prominently feature this exact headline, spelled "
        f"exactly: '{headline}'.",
    ]
    if subtext:
        parts.append(
            f"Include this supporting text verbatim, spelled exactly: '{subtext}'."
        )
    parts.append(
        f"Typography: use {headline_font} for the headline, and {body_font} "
        f"for supporting text. Maintain a cohesive type system across all "
        f"text elements."
    )
    parts.append(ANTI_AI_DESIGN)
    return " ".join(parts)


def _classify_genai_error(
    exc: BaseException, timeout_s: Optional[float] = None
) -> GenerationError:
    if timeout_s is None:
        timeout_s = settings.generation_timeout_s
    """Pure classifier — converts any exception from the genai call site into
    a GenerationError. Centralizing the mapping here makes it unit-testable
    without needing to mock the genai client."""
    if isinstance(exc, asyncio.TimeoutError):
        return GenerationError(
            "timeout",
            f"Generation exceeded {int(timeout_s)}s — Vertex AI is slow today; "
            f"please try again in a moment.",
            504,
        )
    # Lazy import — keeps the module importable even if google-genai isn't installed.
    try:
        from google.genai import errors as genai_errors
        client_err_cls = genai_errors.ClientError
        server_err_cls = genai_errors.ServerError
    except Exception:
        client_err_cls = ()
        server_err_cls = ()

    if client_err_cls and isinstance(exc, client_err_cls):
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status == 429:
            return GenerationError(
                "rate_limit",
                "Vertex AI rate limit reached — try again in a few seconds.",
                429,
            )
        if status == 403:
            return GenerationError(
                "auth_error",
                f"Vertex AI permission denied: {str(exc)[:200]}",
                502,
            )
        return GenerationError(
            "upstream_error",
            f"Vertex AI client error: {str(exc)[:300]}",
            502,
        )
    if server_err_cls and isinstance(exc, server_err_cls):
        return GenerationError(
            "upstream_error",
            f"Vertex AI returned a server error: {str(exc)[:300]}",
            502,
        )
    return GenerationError(
        "unexpected",
        f"{type(exc).__name__}: {str(exc)[:300]}",
        502,
    )


def _classify_finish_reason(finish_reason) -> Optional[GenerationError]:
    """Pure classifier for the response's finish_reason. Returns None on STOP
    (clean completion). Returns a GenerationError on any non-STOP terminal
    reason (safety filter, prohibited content, recitation, etc.)."""
    from google.genai import types

    if finish_reason == types.FinishReason.STOP:
        return None

    blocked_reasons = {
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.PROHIBITED_CONTENT,
        getattr(types.FinishReason, "IMAGE_SAFETY", None),
    }
    blocked_reasons.discard(None)

    if finish_reason in blocked_reasons:
        return GenerationError(
            "content_blocked",
            f"Content blocked by safety policy ({finish_reason.name}). "
            f"Please rephrase your prompt and try again.",
            422,
        )
    return GenerationError(
        "incomplete",
        f"Generation did not complete cleanly (finish_reason={finish_reason.name}).",
        502,
    )


async def generate_design(
    description: str,
    headline: str,
    subtext: Optional[str],
    headline_font: str,
    body_font: str,
    timeout_s: Optional[float] = None,
) -> dict:
    """Async generation. Raises GenerationError on any failure (timeout,
    rate limit, content block, no image, etc.) — caller maps to HTTP."""
    from google.genai import types

    if timeout_s is None:
        timeout_s = settings.generation_timeout_s

    prompt = build_prompt(description, headline, subtext, headline_font, body_font)
    logger.info(
        "generate_design: model=%s, prompt_chars=%d, headline=%r",
        settings.model_id, len(prompt), headline[:60],
    )

    client = _get_client()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=settings.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                        image_size="2K",
                    ),
                ),
            ),
            timeout=timeout_s,
        )
    except GenerationError:
        raise  # already classified somewhere upstream (shouldn't happen but safe)
    except BaseException as exc:
        # asyncio.TimeoutError is BaseException in 3.11+; catching BaseException
        # ensures we map it.
        raise _classify_genai_error(exc, timeout_s) from exc

    if not response.candidates:
        raise GenerationError(
            "no_candidates",
            "Vertex AI returned a response with no candidates.",
            502,
        )

    cand = response.candidates[0]
    finish_err = _classify_finish_reason(cand.finish_reason)
    if finish_err:
        raise finish_err

    image_bytes = None
    inline_mime = None
    reasoning_parts = []
    for part in (cand.content.parts or []):
        if getattr(part, "text", None):
            reasoning_parts.append(part.text)
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            image_bytes = inline.data
            inline_mime = getattr(inline, "mime_type", None)

    if not image_bytes:
        raise GenerationError(
            "no_image",
            "Vertex AI returned a response but it contained no image data. "
            "Please try again or rephrase your prompt.",
            502,
        )

    filename = f"generated_{uuid.uuid4()}.png"
    output_url = _storage.write(filename, image_bytes)
    logger.info(
        "generate_design: saved %s (%d bytes, mime=%s, %d reasoning parts)",
        output_url, len(image_bytes), inline_mime, len(reasoning_parts),
    )

    return {
        "output_path": f"{settings.outputs_dir}/{filename}",
        "output_url": output_url,
        "reasoning_parts": reasoning_parts,
    }
