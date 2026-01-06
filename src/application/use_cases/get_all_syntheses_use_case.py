"""Use case for retrieving all syntheses."""

from typing import List, Optional

from ...domain.repositories.synthesis_repository import SynthesisRepository
from ...domain.repositories.voice_reference_repository import VoiceReferenceRepository
from ..dto.synthesis_dto import SynthesisDTO, VoiceReferenceInfoDTO


class GetAllSynthesesUseCase:
    """
    Use case for retrieving all syntheses with pagination.
    """

    def __init__(
        self,
        synthesis_repository: SynthesisRepository,
        voice_reference_repository: Optional[VoiceReferenceRepository] = None
    ):
        """
        Initialize use case with dependencies.

        Args:
            synthesis_repository: Repository for synthesis persistence.
            voice_reference_repository: Optional repository for voice reference lookup.
        """
        self.synthesis_repo = synthesis_repository
        self.voice_ref_repo = voice_reference_repository

    async def execute(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[SynthesisDTO]:
        """
        Execute the get all syntheses use case.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of SynthesisDTO ordered by creation date (newest first).
        """
        syntheses = await self.synthesis_repo.get_all(limit=limit, offset=offset)

        # Convert to DTOs
        dtos = [SynthesisDTO.from_entity(s) for s in syntheses]

        # Populate voice reference info if repository is available
        if self.voice_ref_repo:
            # Collect unique voice reference IDs
            voice_ref_ids = list(set(
                s.voice_reference_id for s in syntheses
                if s.voice_reference_id is not None
            ))

            if voice_ref_ids:
                # Batch fetch voice references
                voice_refs = await self.voice_ref_repo.get_by_ids(voice_ref_ids)
                voice_ref_map = {vr.id: vr for vr in voice_refs}

                # Populate DTOs with voice reference info
                for dto in dtos:
                    if dto.voice_reference_id and dto.voice_reference_id in voice_ref_map:
                        vr = voice_ref_map[dto.voice_reference_id]
                        dto.voice_reference = VoiceReferenceInfoDTO(
                            id=vr.id,
                            name=vr.name,
                            language=vr.language
                        )

        return dtos

    async def count(self) -> int:
        """
        Get total count of syntheses.

        Returns:
            Total number of synthesis records.
        """
        return await self.synthesis_repo.count()
