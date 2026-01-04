"""Use cases for Chatterbox TTS application layer."""

from .synthesize_text_use_case import SynthesizeTextUseCase
from .get_synthesis_use_case import GetSynthesisUseCase
from .get_all_syntheses_use_case import GetAllSynthesesUseCase
from .delete_synthesis_use_case import DeleteSynthesisUseCase
from .create_voice_reference_use_case import CreateVoiceReferenceUseCase
from .get_voice_reference_use_case import GetVoiceReferenceUseCase
from .get_all_voice_references_use_case import GetAllVoiceReferencesUseCase
from .delete_voice_reference_use_case import DeleteVoiceReferenceUseCase

__all__ = [
    # Synthesis use cases
    "SynthesizeTextUseCase",
    "GetSynthesisUseCase",
    "GetAllSynthesesUseCase",
    "DeleteSynthesisUseCase",
    # Voice reference use cases
    "CreateVoiceReferenceUseCase",
    "GetVoiceReferenceUseCase",
    "GetAllVoiceReferencesUseCase",
    "DeleteVoiceReferenceUseCase",
]
