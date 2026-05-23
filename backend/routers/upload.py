"""Upload router for image upload endpoints"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import uuid
from datetime import datetime

router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 52428800))  # 50MB default


@router.post("/")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for processing
    
    Returns:
        - file_id: Unique identifier for the uploaded file
        - filename: Original filename
        - size: File size in bytes
        - uploaded_at: Timestamp of upload
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Save file
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file_id)
        with open(file_path, "wb") as f:
            f.write(content)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "uploaded_at": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{file_id}")
async def get_upload_status(file_id: str):
    """Get information about an uploaded file"""
    file_path = os.path.join(UPLOAD_DIR, file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    file_stat = os.stat(file_path)
    return {
        "file_id": file_id,
        "size": file_stat.st_size,
        "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat()
    }
