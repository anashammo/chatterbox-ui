"""Use case for retrieving all voice references."""

from typing import List

from ...domain.repositories.voice_reference_repository import VoiceReferenceRepository
from ..dto.voice_reference_dto import VoiceReferenceDTO


class GetAllVoiceReferencesUseCase:
    """
    Use case for retrieving all voice references with pagination.
    """

    def __init__(self, voice_reference_repository: VoiceReferenceRepository):
        """
        Initialize use case with dependencies.

        Args:
            voice_reference_repository: Repository for voice reference persistence.
        """
        self.voice_ref_repo = voice_reference_repository

    async def execute(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[VoiceReferenceDTO]:
        """
        Execute the get all voice references use case.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of VoiceReferenceDTO ordered by creation date (newest first).
        """
        voice_references = await self.voice_ref_repo.get_all(limit=limit, offset=offset)
        return [VoiceReferenceDTO.from_entity(vr) for vr in voice_references]

    async def count(self) -> int:
        """
        Get total count of voice references.

        Returns:
            Total number of voice reference records.
        """
        return await self.voice_ref_repo.count()
