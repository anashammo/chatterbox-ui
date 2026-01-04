"""Domain entities for Chatterbox TTS."""

from .synthesis import Synthesis, SynthesisStatus, TTSModel
from .voice_reference import VoiceReference

__all__ = [
    "Synthesis",
    "SynthesisStatus",
    "TTSModel",
    "VoiceReference",
]
