"""Domain repository interfaces for Chatterbox TTS."""

from .synthesis_repository import SynthesisRepository
from .voice_reference_repository import VoiceReferenceRepository

__all__ = [
    "SynthesisRepository",
    "VoiceReferenceRepository",
]
