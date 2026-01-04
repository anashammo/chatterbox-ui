"""Pydantic schemas for Model API endpoints."""

from typing import List, Optional
from pydantic import BaseModel


class TTSModelResponse(BaseModel):
    """Response body for a TTS model."""

    name: str
    display_name: str
    parameters: str
    description: str
    supports_voice_cloning: bool
    supports_multilingual: bool
    supports_paralinguistics: bool
    is_loaded: bool


class ModelListResponse(BaseModel):
    """Response body for listing available models."""

    models: List[TTSModelResponse]


class ModelStatusResponse(BaseModel):
    """Response body for model status."""

    model_name: str
    is_loaded: bool


class LanguageListResponse(BaseModel):
    """Response body for listing supported languages."""

    languages: List[str]
    count: int


class GPUInfoResponse(BaseModel):
    """Response body for GPU information."""

    available: bool
    device: str
    name: Optional[str] = None
    memory_total_gb: Optional[float] = None
    memory_allocated_gb: Optional[float] = None
    cuda_version: Optional[str] = None
