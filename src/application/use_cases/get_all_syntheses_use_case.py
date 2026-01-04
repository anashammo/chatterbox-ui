"""Use case for retrieving all syntheses."""

from typing import List

from ...domain.repositories.synthesis_repository import SynthesisRepository
from ..dto.synthesis_dto import SynthesisDTO


class GetAllSynthesesUseCase:
    """
    Use case for retrieving all syntheses with pagination.
    """

    def __init__(self, synthesis_repository: SynthesisRepository):
        """
        Initialize use case with dependencies.

        Args:
            synthesis_repository: Repository for synthesis persistence.
        """
        self.synthesis_repo = synthesis_repository

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
        return [SynthesisDTO.from_entity(s) for s in syntheses]

    async def count(self) -> int:
        """
        Get total count of syntheses.

        Returns:
            Total number of synthesis records.
        """
        return await self.synthesis_repo.count()
