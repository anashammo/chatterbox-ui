"""Local filesystem implementation of file storage for TTS."""
import os
import aiofiles
from pathlib import Path
from typing import Optional

from ...application.interfaces.file_storage_interface import FileStorageInterface
from ...domain.exceptions.domain_exception import ServiceException
from ..config.settings import Settings


class LocalFileStorage(FileStorageInterface):
    """
    Local filesystem implementation of file storage interface.

    Stores files in organized directory structure on local disk.
    Used for both audio outputs and voice references.
    """

    def __init__(self, settings: Settings, storage_type: str = "audio_outputs"):
        """
        Initialize local file storage.

        Args:
            settings: Application settings containing storage configuration
            storage_type: Type of storage ("audio_outputs" or "voice_references")
        """
        self.settings = settings
        self.storage_type = storage_type

        if storage_type == "voice_references":
            self.upload_dir = Path(settings.voice_reference_dir)
        else:
            self.upload_dir = Path(settings.audio_output_dir)

        self._ensure_upload_dir()

    def _ensure_upload_dir(self) -> None:
        """Create upload directory if it doesn't exist"""
        try:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ServiceException(f"Failed to create upload directory: {str(e)}")

    def _get_subdirectory(self, file_id: str) -> Path:
        """
        Get subdirectory for file based on ID prefix.

        Uses first 2 characters of ID to create subdirectories
        for better organization when handling many files.

        Args:
            file_id: File identifier

        Returns:
            Path to subdirectory
        """
        subdir_name = file_id[:2] if len(file_id) >= 2 else "00"
        return self.upload_dir / subdir_name

    def _get_file_extension(self, filename: str) -> str:
        """
        Extract file extension from filename.

        Args:
            filename: Original filename

        Returns:
            File extension (including dot) or empty string
        """
        return Path(filename).suffix if filename else ""

    async def save(self, file_content: bytes, file_id: str, filename: str) -> str:
        """
        Save file to local filesystem.

        Args:
            file_content: Binary content of the file
            file_id: Unique identifier for the file
            filename: Original filename (for extension)

        Returns:
            str: Absolute path where file was saved

        Raises:
            ServiceException: If save operation fails
        """
        try:
            # Create subdirectory
            subdir = self._get_subdirectory(file_id)
            subdir.mkdir(parents=True, exist_ok=True)

            # Preserve file extension
            ext = self._get_file_extension(filename)
            file_path = subdir / f"{file_id}{ext}"

            # Save file asynchronously
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)

            return str(file_path.absolute())

        except Exception as e:
            raise ServiceException(f"Failed to save file: {str(e)}")

    async def get(self, file_path: str) -> bytes:
        """
        Retrieve file from local filesystem.

        Args:
            file_path: Path to the file

        Returns:
            bytes: File content

        Raises:
            ServiceException: If file not found or read fails
        """
        try:
            if not os.path.exists(file_path):
                raise ServiceException(f"File not found: {file_path}")

            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()

        except ServiceException:
            raise
        except Exception as e:
            raise ServiceException(f"Failed to read file: {str(e)}")

    async def delete(self, file_path: str) -> bool:
        """
        Delete file from local filesystem.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if deleted, False if not found

        Raises:
            ServiceException: If deletion fails
        """
        try:
            if not os.path.exists(file_path):
                return False

            os.remove(file_path)
            return True

        except Exception as e:
            raise ServiceException(f"Failed to delete file: {str(e)}")

    def exists(self, file_path: str) -> bool:
        """
        Check if file exists in local filesystem.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file exists, False otherwise
        """
        return os.path.exists(file_path)

    def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.

        Args:
            file_path: Path to the file

        Returns:
            int: File size in bytes

        Raises:
            ServiceException: If file not found
        """
        try:
            if not os.path.exists(file_path):
                raise ServiceException(f"File not found: {file_path}")
            return os.path.getsize(file_path)
        except ServiceException:
            raise
        except Exception as e:
            raise ServiceException(f"Failed to get file size: {str(e)}")
