"""Use case for synthesizing text to speech."""

import uuid
from typing import Optional

from ...domain.entities.synthesis import Synthesis, SynthesisStatus, TTSModel
from ...domain.repositories.synthesis_repository import SynthesisRepository
from ...domain.repositories.voice_reference_repository import VoiceReferenceRepository
from ...domain.services.text_to_speech_service import TextToSpeechService
from ...domain.exceptions import ServiceException
from ..dto.synthesis_dto import SynthesisDTO, SynthesisCreateDTO
from ..interfaces.file_storage_interface import FileStorageInterface


class SynthesizeTextUseCase:
    """
    Use case for synthesizing text to speech.

    Orchestrates the TTS synthesis process:
    1. Validates input and creates pending synthesis record
    2. Retrieves voice reference if specified
    3. Calls TTS service for synthesis
    4. Saves output audio file
    5. Updates synthesis record with result
    """

    def __init__(
        self,
        synthesis_repository: SynthesisRepository,
        voice_reference_repository: VoiceReferenceRepository,
        tts_service: TextToSpeechService,
        audio_storage: FileStorageInterface,
    ):
        """
        Initialize use case with dependencies.

        Args:
            synthesis_repository: Repository for synthesis persistence.
            voice_reference_repository: Repository for voice reference lookup.
            tts_service: TTS service implementation.
            audio_storage: Storage for output audio files.
        """
        self.synthesis_repo = synthesis_repository
        self.voice_ref_repo = voice_reference_repository
        self.tts_service = tts_service
        self.audio_storage = audio_storage

    async def execute(self, create_dto: SynthesisCreateDTO) -> SynthesisDTO:
        """
        Execute the synthesis use case.

        Args:
            create_dto: DTO containing synthesis parameters.

        Returns:
            SynthesisDTO with the completed synthesis result.

        Raises:
            ValueError: If input validation fails.
            ServiceException: If synthesis fails.
        """
        # Validate input
        create_dto.validate()

        # Get voice reference path if specified
        voice_reference_path: Optional[str] = None
        if create_dto.voice_reference_id:
            voice_ref = await self.voice_ref_repo.get_by_id(
                create_dto.voice_reference_id
            )
            if voice_ref is None:
                raise ValueError(
                    f"Voice reference not found: {create_dto.voice_reference_id}"
                )
            voice_reference_path = voice_ref.file_path

        # Create synthesis entity
        synthesis_id = str(uuid.uuid4())
        synthesis = Synthesis(
            id=synthesis_id,
            input_text=create_dto.text,
            text_length=len(create_dto.text),
            model=TTSModel(create_dto.model),
            language=create_dto.language,
            voice_reference_id=create_dto.voice_reference_id,
            cfg_weight=create_dto.cfg_weight,
            exaggeration=create_dto.exaggeration,
        )

        # Save pending synthesis
        synthesis = await self.synthesis_repo.create(synthesis)

        # Mark as processing
        synthesis.mark_as_processing()
        synthesis = await self.synthesis_repo.update(synthesis)

        try:
            # Perform synthesis
            result = await self.tts_service.synthesize(
                text=create_dto.text,
                model=create_dto.model,
                voice_reference_path=voice_reference_path,
                language=create_dto.language,
                cfg_weight=create_dto.cfg_weight,
                exaggeration=create_dto.exaggeration,
            )

            # Save audio output
            output_filename = f"{synthesis_id}.wav"
            output_path = await self.audio_storage.save(
                file_content=result["audio_data"],
                file_id=synthesis_id,
                filename=output_filename,
            )

            # Complete synthesis
            synthesis.complete(
                output_file_path=output_path,
                output_duration_seconds=result["duration_seconds"],
                processing_time_seconds=result["processing_time_seconds"],
            )
            synthesis = await self.synthesis_repo.update(synthesis)

        except Exception as e:
            # Mark as failed
            synthesis.fail(str(e))
            await self.synthesis_repo.update(synthesis)
            raise ServiceException(f"Synthesis failed: {str(e)}")

        return SynthesisDTO.from_entity(synthesis)
