from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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


app = FastAPI(title="PixelScript API", version="1.0.0", description="API for AI-generated image font replacement")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

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
    except Exception:
        raise HTTPException(status_code=500, detail="Font replacement dependencies not available")
    regions = detect_text(file_path)
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
