"""Pydantic schemas for Chatterbox TTS API."""

from .synthesis_schema import (
    SynthesisCreateRequest,
    SynthesisResponse,
    SynthesisListResponse,
    SynthesisDeleteResponse,
)
from .voice_reference_schema import (
    VoiceReferenceResponse,
    VoiceReferenceListResponse,
    VoiceReferenceDeleteResponse,
    VoiceReferenceUploadResponse,
)
from .model_schema import (
    TTSModelResponse,
    ModelListResponse,
    ModelStatusResponse,
    LanguageListResponse,
    GPUInfoResponse,
)

__all__ = [
    # Synthesis
    "SynthesisCreateRequest",
    "SynthesisResponse",
    "SynthesisListResponse",
    "SynthesisDeleteResponse",
    # Voice Reference
    "VoiceReferenceResponse",
    "VoiceReferenceListResponse",
    "VoiceReferenceDeleteResponse",
    "VoiceReferenceUploadResponse",
    # Model
    "TTSModelResponse",
    "ModelListResponse",
    "ModelStatusResponse",
    "LanguageListResponse",
    "GPUInfoResponse",
]
