"""Pydantic schemas for Synthesis API endpoints."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class SynthesisCreateRequest(BaseModel):
    """Request body for creating a new synthesis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to synthesize to speech"
    )
    model: str = Field(
        default="turbo",
        description="TTS model to use (turbo, standard, multilingual)"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code for multilingual model (e.g., 'en', 'es', 'fr')"
    )
    voice_reference_id: Optional[str] = Field(
        default=None,
        description="ID of voice reference for voice cloning"
    )
    cfg_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="CFG weight for accent transfer (0.0-1.0)"
    )
    exaggeration: float = Field(
        default=0.5,
        ge=0.0,
        description="Speech expressiveness (0.0-1.0+)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, welcome to Chatterbox TTS!",
                "model": "turbo",
                "voice_reference_id": None,
                "cfg_weight": 0.5,
                "exaggeration": 0.5
            }
        }


class SynthesisResponse(BaseModel):
    """Response body for a synthesis."""

    id: str
    input_text: str
    text_length: int
    model: str
    status: str
    language: Optional[str]
    voice_reference_id: Optional[str]
    cfg_weight: float
    exaggeration: float
    output_file_path: Optional[str]
    output_duration_seconds: Optional[float]
    error_message: Optional[str]
    processing_time_seconds: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class SynthesisListResponse(BaseModel):
    """Response body for listing syntheses."""

    syntheses: List[SynthesisResponse]
    total: int
    limit: int
    offset: int


class SynthesisDeleteResponse(BaseModel):
    """Response body for deleting a synthesis."""

    success: bool
    message: str
    synthesis_id: str
