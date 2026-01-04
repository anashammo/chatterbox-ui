"""Dependency injection container for FastAPI - Chatterbox TTS."""

from functools import lru_cache
from fastapi import Depends
from sqlalchemy.orm import Session

from ...infrastructure.config.settings import Settings, get_settings
from ...infrastructure.persistence.database import get_db
from ...infrastructure.services.chatterbox_service import ChatterboxService
from ...infrastructure.storage.local_file_storage import LocalFileStorage
from ...infrastructure.persistence.repositories.sqlite_synthesis_repository import SQLiteSynthesisRepository
from ...infrastructure.persistence.repositories.sqlite_voice_reference_repository import SQLiteVoiceReferenceRepository
from ...application.use_cases.synthesize_text_use_case import SynthesizeTextUseCase
from ...application.use_cases.get_synthesis_use_case import GetSynthesisUseCase
from ...application.use_cases.get_all_syntheses_use_case import GetAllSynthesesUseCase
from ...application.use_cases.delete_synthesis_use_case import DeleteSynthesisUseCase
from ...application.use_cases.create_voice_reference_use_case import CreateVoiceReferenceUseCase
from ...application.use_cases.get_voice_reference_use_case import GetVoiceReferenceUseCase
from ...application.use_cases.get_all_voice_references_use_case import GetAllVoiceReferencesUseCase
from ...application.use_cases.delete_voice_reference_use_case import DeleteVoiceReferenceUseCase


# Singleton services (loaded once and reused)
@lru_cache()
def get_tts_service() -> ChatterboxService:
    """
    Get Chatterbox TTS service singleton.

    TTS models are loaded lazily on first use and reused
    across requests for performance.

    Returns:
        ChatterboxService instance
    """
    settings = get_settings()
    return ChatterboxService(settings)


@lru_cache()
def get_audio_output_storage() -> LocalFileStorage:
    """
    Get audio output storage service singleton.

    Used for storing synthesized audio files.

    Returns:
        LocalFileStorage instance for audio outputs
    """
    settings = get_settings()
    return LocalFileStorage(settings, storage_type="audio_outputs")


@lru_cache()
def get_voice_reference_storage() -> LocalFileStorage:
    """
    Get voice reference storage service singleton.

    Used for storing voice reference audio files for cloning.

    Returns:
        LocalFileStorage instance for voice references
    """
    settings = get_settings()
    return LocalFileStorage(settings, storage_type="voice_references")


# Use case factory functions with dependency injection

def get_synthesize_text_use_case(
    db: Session = Depends(get_db),
    tts_service: ChatterboxService = Depends(get_tts_service),
    audio_storage: LocalFileStorage = Depends(get_audio_output_storage),
) -> SynthesizeTextUseCase:
    """
    Create SynthesizeTextUseCase with all dependencies injected.

    Args:
        db: Database session
        tts_service: Chatterbox TTS service
        audio_storage: Audio output storage service

    Returns:
        SynthesizeTextUseCase instance
    """
    synthesis_repo = SQLiteSynthesisRepository(db)
    voice_ref_repo = SQLiteVoiceReferenceRepository(db)

    return SynthesizeTextUseCase(
        synthesis_repository=synthesis_repo,
        voice_reference_repository=voice_ref_repo,
        tts_service=tts_service,
        audio_storage=audio_storage,
    )


def get_synthesis_use_case(
    db: Session = Depends(get_db)
) -> GetSynthesisUseCase:
    """
    Create GetSynthesisUseCase with dependencies injected.

    Args:
        db: Database session

    Returns:
        GetSynthesisUseCase instance
    """
    synthesis_repo = SQLiteSynthesisRepository(db)
    return GetSynthesisUseCase(synthesis_repo)


def get_all_syntheses_use_case(
    db: Session = Depends(get_db)
) -> GetAllSynthesesUseCase:
    """
    Create GetAllSynthesesUseCase with dependencies injected.

    Args:
        db: Database session

    Returns:
        GetAllSynthesesUseCase instance
    """
    synthesis_repo = SQLiteSynthesisRepository(db)
    return GetAllSynthesesUseCase(synthesis_repo)


def get_delete_synthesis_use_case(
    db: Session = Depends(get_db),
    audio_storage: LocalFileStorage = Depends(get_audio_output_storage)
) -> DeleteSynthesisUseCase:
    """
    Create DeleteSynthesisUseCase with dependencies injected.

    Args:
        db: Database session
        audio_storage: Audio output storage service

    Returns:
        DeleteSynthesisUseCase instance
    """
    synthesis_repo = SQLiteSynthesisRepository(db)
    return DeleteSynthesisUseCase(
        synthesis_repository=synthesis_repo,
        audio_storage=audio_storage
    )


def get_create_voice_reference_use_case(
    db: Session = Depends(get_db),
    voice_storage: LocalFileStorage = Depends(get_voice_reference_storage)
) -> CreateVoiceReferenceUseCase:
    """
    Create CreateVoiceReferenceUseCase with dependencies injected.

    Args:
        db: Database session
        voice_storage: Voice reference storage service

    Returns:
        CreateVoiceReferenceUseCase instance
    """
    voice_ref_repo = SQLiteVoiceReferenceRepository(db)
    return CreateVoiceReferenceUseCase(
        voice_reference_repository=voice_ref_repo,
        voice_storage=voice_storage
    )


def get_voice_reference_use_case(
    db: Session = Depends(get_db)
) -> GetVoiceReferenceUseCase:
    """
    Create GetVoiceReferenceUseCase with dependencies injected.

    Args:
        db: Database session

    Returns:
        GetVoiceReferenceUseCase instance
    """
    voice_ref_repo = SQLiteVoiceReferenceRepository(db)
    return GetVoiceReferenceUseCase(voice_ref_repo)


def get_all_voice_references_use_case(
    db: Session = Depends(get_db)
) -> GetAllVoiceReferencesUseCase:
    """
    Create GetAllVoiceReferencesUseCase with dependencies injected.

    Args:
        db: Database session

    Returns:
        GetAllVoiceReferencesUseCase instance
    """
    voice_ref_repo = SQLiteVoiceReferenceRepository(db)
    return GetAllVoiceReferencesUseCase(voice_ref_repo)


def get_delete_voice_reference_use_case(
    db: Session = Depends(get_db),
    voice_storage: LocalFileStorage = Depends(get_voice_reference_storage)
) -> DeleteVoiceReferenceUseCase:
    """
    Create DeleteVoiceReferenceUseCase with dependencies injected.

    Args:
        db: Database session
        voice_storage: Voice reference storage service

    Returns:
        DeleteVoiceReferenceUseCase instance
    """
    voice_ref_repo = SQLiteVoiceReferenceRepository(db)
    synthesis_repo = SQLiteSynthesisRepository(db)
    return DeleteVoiceReferenceUseCase(
        voice_reference_repository=voice_ref_repo,
        synthesis_repository=synthesis_repo,
        voice_storage=voice_storage
    )
