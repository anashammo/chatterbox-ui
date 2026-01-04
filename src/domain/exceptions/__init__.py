"""Domain layer exceptions for Chatterbox TTS."""

from .domain_exception import (
    DomainException,
    ValidationException,
    RepositoryException,
    SynthesisException,
    VoiceReferenceException,
    ServiceException,
)

__all__ = [
    "DomainException",
    "ValidationException",
    "RepositoryException",
    "SynthesisException",
    "VoiceReferenceException",
    "ServiceException",
]
