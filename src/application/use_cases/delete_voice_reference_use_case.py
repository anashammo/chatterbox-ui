"""Use case for deleting a voice reference."""

from ...domain.repositories.voice_reference_repository import VoiceReferenceRepository
from ...domain.repositories.synthesis_repository import SynthesisRepository
from ..interfaces.file_storage_interface import FileStorageInterface


class DeleteVoiceReferenceUseCase:
    """
    Use case for deleting a voice reference and its audio file.

    Note: Syntheses that used this voice reference will have their
    voice_reference_id set to NULL (not deleted).
    """

    def __init__(
        self,
        voice_reference_repository: VoiceReferenceRepository,
        synthesis_repository: SynthesisRepository,
        voice_storage: FileStorageInterface,
    ):
        """
        Initialize use case with dependencies.

        Args:
            voice_reference_repository: Repository for voice reference persistence.
            synthesis_repository: Repository for synthesis persistence.
            voice_storage: Storage for voice reference audio files.
        """
        self.voice_ref_repo = voice_reference_repository
        self.synthesis_repo = synthesis_repository
        self.voice_storage = voice_storage

    async def execute(self, voice_reference_id: str) -> bool:
        """
        Execute the delete voice reference use case.

        Deletes both the database record and the audio file.
        Syntheses using this voice reference will have voice_reference_id set to NULL.

        Args:
            voice_reference_id: The unique identifier of the voice reference to delete.

        Returns:
            True if deleted successfully, False if not found.
        """
        # Get voice reference to find the file path
        voice_reference = await self.voice_ref_repo.get_by_id(voice_reference_id)

        if voice_reference is None:
            return False

        # Delete the audio file
        if voice_reference.file_path:
            await self.voice_storage.delete(voice_reference.file_path)

        # Delete database record (FK constraint with SET NULL handles syntheses)
        return await self.voice_ref_repo.delete(voice_reference_id)

    async def get_usage_count(self, voice_reference_id: str) -> int:
        """
        Get count of syntheses using this voice reference.

        Args:
            voice_reference_id: The voice reference ID to check.

        Returns:
            Number of syntheses using this voice reference.
        """
        syntheses = await self.synthesis_repo.get_by_voice_reference_id(voice_reference_id)
        return len(syntheses)
