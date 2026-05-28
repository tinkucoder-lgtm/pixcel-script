from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
import base64
import json
import logging
import mimetypes
import os
import shutil
import uuid
import requests
# Lazy imports for heavy image-processing modules to avoid import errors
# at server startup when optional dependencies aren't installed.

from config.limiter import limiter
from config.settings import settings


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(_app):
    # Startup: sweep old output files. Cheap; iterates a few hundred entries.
    from services.storage import LocalStorage
    storage = LocalStorage(settings.outputs_dir)
    deleted = storage.cleanup_older_than(settings.output_cleanup_age_hours)
    logging.info(
        "startup_cleanup: deleted %d output files older than %d hours",
        deleted, settings.output_cleanup_age_hours,
    )
    yield
    # Shutdown: nothing to do.


app = FastAPI(
    title="PixelScript API",
    version="1.0.0",
    description="API for AI-generated image font replacement + design generation",
    lifespan=lifespan,
)

# CORS: tight by default (configurable via PIXELSCRIPT_CORS_ALLOWED_ORIGINS).
# NEVER set this to ["*"] in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (slowapi). The limiter itself is shared in config/limiter.py
# so router decorators and this handler use the same instance.
app.state.limiter = limiter


def _rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded ({exc.detail}). Try again shortly.",
        },
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

UPLOADS_DIR = "uploads"
OUTPUTS_DIR = settings.outputs_dir
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Routers — new features live here per project convention; the inline endpoints
# below predate this pattern and stay where they are to avoid churn.
from routers import generate_design
app.include_router(generate_design.router)


@app.get("/api/health")
async def api_health():
    """Cheap liveness + config introspection. Initializes the Vertex client
    if not already (idempotent; cached singleton) but does NOT call Gemini."""
    try:
        from services.design_generator import _get_client
        _get_client()
        return {
            "status": "ok",
            "model": settings.model_id,
            "project": settings.gcp_project,
            "location": settings.vertex_location,
            "rate_limit": settings.rate_limit_str,
            "timeout_s": settings.generation_timeout_s,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)[:200]},
        )

class ProcessRequest(BaseModel):
    file_id: str

class FontRequest(BaseModel):
    file_id: str
    font_name: str

class HumanizeRequest(BaseModel):
    file_id: str
    style: str



@app.get("/")
def root():
    return {"message": "PixelScript API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".png"
    file_path = os.path.join(UPLOADS_DIR, f"{file_id}{ext}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"file_id": file_id, "filename": file.filename, "path": file_path}

@app.post("/api/process")
async def process_image(req: ProcessRequest):
    matches = [f for f in os.listdir(UPLOADS_DIR) if f.startswith(req.file_id)]
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = os.path.join(UPLOADS_DIR, matches[0])
    try:
        from ocr import detect_text
    except Exception:
        raise HTTPException(status_code=500, detail="OCR dependency not available")
    regions = detect_text(file_path)
    return {"file_id": req.file_id, "regions": regions, "count": len(regions)}

@app.post("/api/replace-font")
async def replace_font(req: FontRequest):
    matches = [f for f in os.listdir(UPLOADS_DIR) if f.startswith(req.file_id)]
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = os.path.join(UPLOADS_DIR, matches[0])
    try:
        from ocr import detect_text
        from font_replacer import replace_fonts
        from region_validator import validate_legibility
        # Disabled — caused inpaint-mask ghosting on dense images; re-enable with v2 expansion fix
        # from region_normalizer import normalize_fontsize_groups
    except Exception:
        raise HTTPException(status_code=500, detail="Font replacement dependencies not available")
    regions = detect_text(file_path)
    regions = validate_legibility(regions, req.font_name)
    # Disabled — caused inpaint-mask ghosting on dense images; re-enable with v2 expansion fix
    # regions = normalize_fontsize_groups(regions, req.font_name)
    output_filename = f"{req.file_id}_output.png"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    replace_fonts(file_path, regions, req.font_name, output_path)
    return {"file_id": req.file_id, "font": req.font_name, "output_url": f"/outputs/{output_filename}"}

@app.post("/api/humanize")
async def humanize_image(req: HumanizeRequest):
    import cv2
    from humanizer import apply_watercolor, apply_sketch, apply_oil_painting, apply_flat_art, apply_vintage

    output_path = os.path.join(OUTPUTS_DIR, f"{req.file_id}_output.png")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Font-replaced output not found")

    img = cv2.imread(output_path)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load image")

    style_map = {
        "watercolor": apply_watercolor,
        "sketch": apply_sketch,
        "oil_painting": apply_oil_painting,
        "flat_art": apply_flat_art,
        "vintage": apply_vintage,
    }
    fn = style_map.get(req.style)
    if fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown style: {req.style}")

    result = fn(img)
    styled_filename = f"{req.file_id}_{req.style}.png"
    styled_path = os.path.join(OUTPUTS_DIR, styled_filename)
    cv2.imwrite(styled_path, result)

    return {"output_url": f"/outputs/{styled_filename}"}

@app.get("/api/status/{file_id}")
def get_status(file_id: str):
    matches = [f for f in os.listdir(UPLOADS_DIR) if f.startswith(file_id)]
    return {"file_id": file_id, "exists": len(matches) > 0}
