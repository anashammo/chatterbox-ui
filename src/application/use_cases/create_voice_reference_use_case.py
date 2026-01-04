"""Use case for creating a voice reference."""

import uuid

from ...domain.entities.voice_reference import VoiceReference
from ...domain.repositories.voice_reference_repository import VoiceReferenceRepository
from ...domain.exceptions import ServiceException
from ..dto.voice_reference_dto import VoiceReferenceDTO, VoiceReferenceCreateDTO
from ..interfaces.file_storage_interface import FileStorageInterface


class CreateVoiceReferenceUseCase:
    """
    Use case for creating a new voice reference for voice cloning.

    Handles:
    1. Validating the uploaded audio file
    2. Saving the audio file to storage
    3. Creating the voice reference record
    """

    def __init__(
        self,
        voice_reference_repository: VoiceReferenceRepository,
        voice_storage: FileStorageInterface,
    ):
        """
        Initialize use case with dependencies.

        Args:
            voice_reference_repository: Repository for voice reference persistence.
            voice_storage: Storage for voice reference audio files.
        """
        self.voice_ref_repo = voice_reference_repository
        self.voice_storage = voice_storage

    async def execute(self, create_dto: VoiceReferenceCreateDTO) -> VoiceReferenceDTO:
        """
        Execute the create voice reference use case.

        Args:
            create_dto: DTO containing voice reference data.

        Returns:
            VoiceReferenceDTO with the created voice reference.

        Raises:
            ValueError: If validation fails.
            ServiceException: If creation fails.
        """
        # Validate input
        create_dto.validate()

        # Check if name already exists
        existing = await self.voice_ref_repo.get_by_name(create_dto.name)
        if existing:
            raise ValueError(f"Voice reference with name '{create_dto.name}' already exists")

        # Generate ID and save file
        voice_ref_id = str(uuid.uuid4())

        try:
            file_path = await self.voice_storage.save(
                file_content=create_dto.file_content,
                file_id=voice_ref_id,
                filename=create_dto.original_filename,
            )

            # Create voice reference entity
            voice_reference = VoiceReference(
                id=voice_ref_id,
                name=create_dto.name,
                original_filename=create_dto.original_filename,
                file_path=file_path,
                file_size_bytes=len(create_dto.file_content),
                mime_type=create_dto.mime_type,
                duration_seconds=create_dto.duration_seconds,
            )

            # Save to repository
            voice_reference = await self.voice_ref_repo.create(voice_reference)

            return VoiceReferenceDTO.from_entity(voice_reference)

        except Exception as e:
            # Clean up file if database save failed
            try:
                await self.voice_storage.delete(file_path)
            except Exception:
                pass
            raise ServiceException(f"Failed to create voice reference: {str(e)}")
