"""Use case for retrieving a voice reference by ID."""

from typing import Optional

from ...domain.repositories.voice_reference_repository import VoiceReferenceRepository
from ..dto.voice_reference_dto import VoiceReferenceDTO


class GetVoiceReferenceUseCase:
    """
    Use case for retrieving a single voice reference by ID.
    """

    def __init__(self, voice_reference_repository: VoiceReferenceRepository):
        """
        Initialize use case with dependencies.

        Args:
            voice_reference_repository: Repository for voice reference persistence.
        """
        self.voice_ref_repo = voice_reference_repository

    async def execute(self, voice_reference_id: str) -> Optional[VoiceReferenceDTO]:
        """
        Execute the get voice reference use case.

        Args:
            voice_reference_id: The unique identifier of the voice reference.

        Returns:
            VoiceReferenceDTO if found, None otherwise.
        """
        voice_reference = await self.voice_ref_repo.get_by_id(voice_reference_id)

        if voice_reference is None:
            return None

        return VoiceReferenceDTO.from_entity(voice_reference)
