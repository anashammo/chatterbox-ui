"""Synthesis API endpoints for Chatterbox TTS."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Optional

from ..schemas.synthesis_schema import (
    SynthesisCreateRequest,
    SynthesisResponse,
    SynthesisListResponse,
    SynthesisDeleteResponse,
)
from ..dependencies import (
    get_synthesize_text_use_case,
    get_synthesis_use_case,
    get_all_syntheses_use_case,
    get_delete_synthesis_use_case,
)
from ....application.use_cases.synthesize_text_use_case import SynthesizeTextUseCase
from ....application.use_cases.get_synthesis_use_case import GetSynthesisUseCase
from ....application.use_cases.get_all_syntheses_use_case import GetAllSynthesesUseCase
from ....application.use_cases.delete_synthesis_use_case import DeleteSynthesisUseCase
from ....application.dto.synthesis_dto import SynthesisCreateDTO
from ....domain.exceptions import ServiceException

router = APIRouter(prefix="/syntheses", tags=["syntheses"])


@router.post("", response_model=SynthesisResponse, status_code=201)
async def create_synthesis(
    request: SynthesisCreateRequest,
    use_case: SynthesizeTextUseCase = Depends(get_synthesize_text_use_case),
) -> SynthesisResponse:
    """
    Create a new TTS synthesis.

    Synthesizes the provided text to speech using the specified model.
    Optionally uses a voice reference for voice cloning.

    Returns the completed synthesis with output audio file path.
    """
    try:
        create_dto = SynthesisCreateDTO(
            text=request.text,
            model=request.model,
            language=request.language,
            voice_reference_id=request.voice_reference_id,
            cfg_weight=request.cfg_weight,
            exaggeration=request.exaggeration,
        )

        result = await use_case.execute(create_dto)

        return SynthesisResponse(**result.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceException as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=SynthesisListResponse)
async def list_syntheses(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    use_case: GetAllSynthesesUseCase = Depends(get_all_syntheses_use_case),
) -> SynthesisListResponse:
    """
    List all syntheses with pagination.

    Returns syntheses ordered by creation date (newest first).
    """
    try:
        syntheses = await use_case.execute(limit=limit, offset=offset)
        total = await use_case.count()

        return SynthesisListResponse(
            syntheses=[SynthesisResponse(**s.to_dict()) for s in syntheses],
            total=total,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{synthesis_id}", response_model=SynthesisResponse)
async def get_synthesis(
    synthesis_id: str,
    use_case: GetSynthesisUseCase = Depends(get_synthesis_use_case),
) -> SynthesisResponse:
    """
    Get a synthesis by ID.

    Returns the synthesis details including status and output file path.
    """
    result = await use_case.execute(synthesis_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Synthesis not found")

    return SynthesisResponse(**result.to_dict())


@router.delete("/{synthesis_id}", response_model=SynthesisDeleteResponse)
async def delete_synthesis(
    synthesis_id: str,
    use_case: DeleteSynthesisUseCase = Depends(get_delete_synthesis_use_case),
) -> SynthesisDeleteResponse:
    """
    Delete a synthesis and its output audio file.

    Permanently removes the synthesis record and associated audio file.
    """
    try:
        success = await use_case.execute(synthesis_id)

        if not success:
            raise HTTPException(status_code=404, detail="Synthesis not found")

        return SynthesisDeleteResponse(
            success=True,
            message="Synthesis deleted successfully",
            synthesis_id=synthesis_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{synthesis_id}/audio")
async def get_synthesis_audio(
    synthesis_id: str,
    download: bool = Query(default=False, description="Force browser download"),
    use_case: GetSynthesisUseCase = Depends(get_synthesis_use_case),
):
    """
    Get the output audio file for a synthesis.

    Streams the audio file for playback or triggers a download.
    """
    result = await use_case.execute(synthesis_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Synthesis not found")

    if result.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Synthesis not completed. Current status: {result.status}"
        )

    if not result.output_file_path:
        raise HTTPException(status_code=404, detail="Audio file not found")

    headers = {}
    if download:
        filename = f"synthesis_{synthesis_id}.wav"
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return FileResponse(
        path=result.output_file_path,
        media_type="audio/wav",
        filename=f"synthesis_{synthesis_id}.wav",
        headers=headers,
    )
