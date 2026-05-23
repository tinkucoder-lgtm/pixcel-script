from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    """Response schema for image upload"""
    file_id: str
    filename: str
    size: int
    uploaded_at: str


class ProcessRequest(BaseModel):
    """Request schema for image processing"""
    file_id: str
    font_name: Optional[str] = None
    processing_options: Optional[dict] = None


class ProcessResponse(BaseModel):
    """Response schema for image processing"""
    file_id: str
    status: str
    progress: int
    result_url: Optional[str] = None
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """Response schema for status check"""
    file_id: str
    status: str
    progress: int
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
