"""Data Transfer Objects for Chatterbox TTS application layer."""

from .synthesis_dto import SynthesisDTO, SynthesisCreateDTO
from .voice_reference_dto import VoiceReferenceDTO, VoiceReferenceCreateDTO

__all__ = [
    "SynthesisDTO",
    "SynthesisCreateDTO",
    "VoiceReferenceDTO",
    "VoiceReferenceCreateDTO",
]
