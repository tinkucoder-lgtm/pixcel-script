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


def parse_humanize_response(content_type: str, data: bytes):
    if content_type.startswith("image/"):
        return data

    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=502, detail="Unexpected humanize server response")

    if isinstance(payload, dict):
        if "image_base64" in payload:
            return base64.b64decode(payload["image_base64"])
        if "output" in payload:
            output = payload["output"]
            if isinstance(output, str) and output.startswith("data:image/"):
                header, _, b64 = output.partition(",")
                return base64.b64decode(b64)
            if isinstance(output, str):
                return base64.b64decode(output)
        if "image" in payload:
            image_data = payload["image"]
            if isinstance(image_data, str):
                if image_data.startswith("data:image/"):
                    _, _, b64 = image_data.partition(",")
                    return base64.b64decode(b64)
                return base64.b64decode(image_data)

    raise HTTPException(status_code=502, detail="Unexpected humanize server response")

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
    logging.info("Received humanize request for file_id=%s style=%s", req.file_id, req.style)
    output_filename = f"{req.file_id}_output.png"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    if os.path.exists(output_path):
        image_path = output_path
        logging.info("Using font-replaced image for humanize: %s", output_path)
    else:
        matches = [f for f in os.listdir(UPLOADS_DIR) if f.startswith(req.file_id)]
        if not matches:
            logging.error("No font-replaced or uploaded image found for file_id=%s", req.file_id)
            raise HTTPException(status_code=404, detail="No image found for humanize request")
        image_path = os.path.join(UPLOADS_DIR, matches[0])
        logging.info("Fallback to original uploaded image for humanize: %s", image_path)

    try:
        with open(image_path, "rb") as f:
            file_bytes = f.read()
    except Exception as exc:
        logging.exception("Failed to read image for humanize: %s", image_path)
        raise HTTPException(status_code=500, detail="Unable to read input image")

    files = {
        "file": (os.path.basename(image_path), file_bytes, mimetypes.guess_type(image_path)[0] or "application/octet-stream")
    }
    data = {"style": req.style, "strength": 0.45}

    try:
        response = requests.post(
            "https://armband-washing-morality.ngrok-free.dev/humanize",
            files=files,
            data=data,
            timeout=120,
        )
    except requests.exceptions.RequestException as exc:
        logging.exception("Humanize server request failed")
        raise HTTPException(status_code=502, detail="Humanize server request failed")

    if response.status_code != 200:
        logging.error("Humanize server returned %s: %s", response.status_code, response.text[:500])
        raise HTTPException(status_code=502, detail=f"Humanize server returned {response.status_code}")

    try:
        image_bytes = parse_humanize_response(response.headers.get("Content-Type", ""), response.content)
    except HTTPException:
        raise
    except Exception:
        logging.exception("Failed to parse humanize response")
        raise HTTPException(status_code=502, detail="Invalid response from humanize server")

    humanized_filename = f"{req.file_id}_humanized.png"
    humanized_path = os.path.join(OUTPUTS_DIR, humanized_filename)
    try:
        with open(humanized_path, "wb") as f:
            f.write(image_bytes)
    except Exception as exc:
        logging.exception("Failed to save humanized image: %s", humanized_path)
        raise HTTPException(status_code=500, detail="Unable to save humanized image")

    logging.info("Humanized image saved: %s", humanized_path)
    return {"output_url": f"/outputs/{humanized_filename}"}

@app.get("/api/status/{file_id}")
def get_status(file_id: str):
    matches = [f for f in os.listdir(UPLOADS_DIR) if f.startswith(file_id)]
    return {"file_id": file_id, "exists": len(matches) > 0}
