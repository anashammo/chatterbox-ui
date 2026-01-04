"""Pydantic schemas for Voice Reference API endpoints."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class VoiceReferenceResponse(BaseModel):
    """Response body for a voice reference."""

    id: str
    name: str
    original_filename: str
    file_size_bytes: int
    file_size_mb: float
    mime_type: str
    duration_seconds: float
    created_at: datetime

    class Config:
        from_attributes = True


class VoiceReferenceListResponse(BaseModel):
    """Response body for listing voice references."""

    voice_references: List[VoiceReferenceResponse]
    total: int
    limit: int
    offset: int


class VoiceReferenceDeleteResponse(BaseModel):
    """Response body for deleting a voice reference."""

    success: bool
    message: str
    voice_reference_id: str
    syntheses_affected: int = Field(
        description="Number of syntheses that used this voice reference (now orphaned)"
    )


class VoiceReferenceUploadResponse(BaseModel):
    """Response body after uploading a voice reference."""

    id: str
    name: str
    original_filename: str
    file_size_mb: float
    duration_seconds: float
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "abc123-def456",
                "name": "My Voice",
                "original_filename": "voice_sample.wav",
                "file_size_mb": 0.5,
                "duration_seconds": 10.5,
                "message": "Voice reference uploaded successfully"
            }
        }
