"""Voice Reference API endpoints for Chatterbox TTS."""

import io
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
import soundfile as sf

from ..schemas.voice_reference_schema import (
    VoiceReferenceResponse,
    VoiceReferenceListResponse,
    VoiceReferenceDeleteResponse,
    VoiceReferenceUploadResponse,
)
from ..dependencies import (
    get_create_voice_reference_use_case,
    get_voice_reference_use_case,
    get_all_voice_references_use_case,
    get_delete_voice_reference_use_case,
)
from ....application.use_cases.create_voice_reference_use_case import CreateVoiceReferenceUseCase
from ....application.use_cases.get_voice_reference_use_case import GetVoiceReferenceUseCase
from ....application.use_cases.get_all_voice_references_use_case import GetAllVoiceReferencesUseCase
from ....application.use_cases.delete_voice_reference_use_case import DeleteVoiceReferenceUseCase
from ....application.dto.voice_reference_dto import VoiceReferenceCreateDTO
from ....domain.exceptions import ServiceException

router = APIRouter(prefix="/voice-references", tags=["voice-references"])


def _get_audio_duration_sync(file_content: bytes) -> float:
    """
    Get duration of audio file in seconds (synchronous).

    This is a blocking I/O operation that should be run in a thread pool.

    Args:
        file_content: Raw audio bytes.

    Returns:
        Duration in seconds.
    """
    audio_buffer = io.BytesIO(file_content)
    # Use soundfile to read audio (more reliable than torchaudio)
    audio, sample_rate = sf.read(audio_buffer)
    # Handle mono/stereo: audio shape is (samples,) or (samples, channels)
    num_samples = audio.shape[0]
    return num_samples / sample_rate


async def get_audio_duration(file_content: bytes, mime_type: str) -> float:
    """
    Get duration of audio file in seconds (async).

    Runs the blocking audio read operation in a thread pool to avoid
    blocking the event loop.

    Args:
        file_content: Raw audio bytes.
        mime_type: MIME type of the audio.

    Returns:
        Duration in seconds.
    """
    try:
        # Run blocking I/O in thread pool
        return await asyncio.to_thread(_get_audio_duration_sync, file_content)
    except Exception as e:
        raise ValueError(f"Failed to read audio file: {str(e)}")


@router.post("", response_model=VoiceReferenceUploadResponse, status_code=201)
async def upload_voice_reference(
    file: UploadFile = File(..., description="Voice reference audio file (5-30 seconds)"),
    name: str = Form(..., min_length=1, max_length=255, description="Name for this voice"),
    use_case: CreateVoiceReferenceUseCase = Depends(get_create_voice_reference_use_case),
) -> VoiceReferenceUploadResponse:
    """
    Upload a voice reference for voice cloning.

    The audio should be 5-30 seconds (ideally ~10 seconds) of clear speech.
    Supported formats: WAV, MP3, FLAC, OGG.
    Maximum file size: 10MB.
    """
    try:
        # Read file content
        file_content = await file.read()

        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Validate file size (10MB max)
        max_size = 10 * 1024 * 1024
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is 10MB, got {len(file_content) / (1024 * 1024):.1f}MB"
            )

        # Validate MIME type
        mime_type = file.content_type or "application/octet-stream"
        supported_types = {
            "audio/wav", "audio/x-wav",
            "audio/mpeg", "audio/mp3",
            "audio/flac", "audio/ogg",
        }

        if mime_type not in supported_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio type: {mime_type}. Supported: WAV, MP3, FLAC, OGG"
            )

        # Get audio duration (runs in thread pool to avoid blocking)
        try:
            duration_seconds = await get_audio_duration(file_content, mime_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create DTO
        create_dto = VoiceReferenceCreateDTO(
            name=name,
            original_filename=file.filename or "voice_reference",
            file_content=file_content,
            mime_type=mime_type,
            duration_seconds=duration_seconds,
        )

        # Validate and create
        create_dto.validate()
        result = await use_case.execute(create_dto)

        return VoiceReferenceUploadResponse(
            id=result.id,
            name=result.name,
            original_filename=result.original_filename,
            file_size_mb=round(result.file_size_bytes / (1024 * 1024), 2),
            duration_seconds=result.duration_seconds,
            message="Voice reference uploaded successfully",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("", response_model=VoiceReferenceListResponse)
async def list_voice_references(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    use_case: GetAllVoiceReferencesUseCase = Depends(get_all_voice_references_use_case),
) -> VoiceReferenceListResponse:
    """
    List all voice references with pagination.

    Returns voice references ordered by creation date (newest first).
    """
    try:
        voice_refs = await use_case.execute(limit=limit, offset=offset)
        total = await use_case.count()

        return VoiceReferenceListResponse(
            voice_references=[
                VoiceReferenceResponse(
                    id=vr.id,
                    name=vr.name,
                    original_filename=vr.original_filename,
                    file_size_bytes=vr.file_size_bytes,
                    file_size_mb=round(vr.file_size_bytes / (1024 * 1024), 2),
                    mime_type=vr.mime_type,
                    duration_seconds=vr.duration_seconds,
                    created_at=vr.created_at,
                )
                for vr in voice_refs
            ],
            total=total,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{voice_reference_id}", response_model=VoiceReferenceResponse)
async def get_voice_reference(
    voice_reference_id: str,
    use_case: GetVoiceReferenceUseCase = Depends(get_voice_reference_use_case),
) -> VoiceReferenceResponse:
    """
    Get a voice reference by ID.

    Returns the voice reference details.
    """
    result = await use_case.execute(voice_reference_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Voice reference not found")

    return VoiceReferenceResponse(
        id=result.id,
        name=result.name,
        original_filename=result.original_filename,
        file_size_bytes=result.file_size_bytes,
        file_size_mb=round(result.file_size_bytes / (1024 * 1024), 2),
        mime_type=result.mime_type,
        duration_seconds=result.duration_seconds,
        created_at=result.created_at,
    )


@router.delete("/{voice_reference_id}", response_model=VoiceReferenceDeleteResponse)
async def delete_voice_reference(
    voice_reference_id: str,
    use_case: DeleteVoiceReferenceUseCase = Depends(get_delete_voice_reference_use_case),
) -> VoiceReferenceDeleteResponse:
    """
    Delete a voice reference and its audio file.

    Syntheses that used this voice reference will have their
    voice_reference_id set to NULL (they are not deleted).
    """
    try:
        # Get usage count before deletion
        usage_count = await use_case.get_usage_count(voice_reference_id)

        success = await use_case.execute(voice_reference_id)

        if not success:
            raise HTTPException(status_code=404, detail="Voice reference not found")

        return VoiceReferenceDeleteResponse(
            success=True,
            message="Voice reference deleted successfully",
            voice_reference_id=voice_reference_id,
            syntheses_affected=usage_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{voice_reference_id}/audio")
async def get_voice_reference_audio(
    voice_reference_id: str,
    download: bool = Query(default=False, description="Force browser download"),
    use_case: GetVoiceReferenceUseCase = Depends(get_voice_reference_use_case),
):
    """
    Get the audio file for a voice reference.

    Streams the audio file for playback or triggers a download.
    """
    result = await use_case.execute(voice_reference_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Voice reference not found")

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{result.original_filename}"'

    return FileResponse(
        path=result.file_path,
        media_type=result.mime_type,
        filename=result.original_filename,
        headers=headers,
    )
