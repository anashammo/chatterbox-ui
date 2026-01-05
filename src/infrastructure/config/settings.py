"""Application settings using Pydantic for configuration management"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Uses Pydantic for validation and type conversion.
    Settings can be overridden via .env file or environment variables.
    """

    # Application
    app_name: str = Field(default="Chatterbox TTS API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8002, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API prefix")

    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:4201", "http://localhost:4200", "http://localhost:3000", "*"],
        description="Allowed CORS origins"
    )

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./chatterbox_tts.db",
        description="Database connection URL"
    )

    # File Storage Configuration
    audio_output_dir: str = Field(
        default="./audio_outputs",
        description="Directory for synthesized audio output files"
    )
    voice_reference_dir: str = Field(
        default="./voice_references",
        description="Directory for voice reference audio files"
    )
    max_voice_reference_size_mb: int = Field(
        default=10,
        description="Maximum voice reference file size in MB"
    )
    max_voice_reference_duration_seconds: int = Field(
        default=30,
        description="Maximum voice reference audio duration in seconds"
    )
    min_voice_reference_duration_seconds: int = Field(
        default=5,
        description="Minimum voice reference audio duration in seconds"
    )

    # Chatterbox TTS Configuration
    tts_default_model: str = Field(
        default="turbo",
        description="Default TTS model (turbo, standard, multilingual)"
    )
    tts_device: str = Field(
        default="cuda",
        description="Device for TTS inference (cuda or cpu)"
    )
    tts_default_cfg_weight: float = Field(
        default=0.5,
        description="Default CFG weight for accent transfer (0.0-1.0)"
    )
    tts_default_exaggeration: float = Field(
        default=0.5,
        description="Default exaggeration/expressiveness (0.0-1.0+)"
    )
    tts_max_text_length: int = Field(
        default=5000,
        description="Maximum text length for synthesis in characters"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")

    class Config:
        # Load environment from backend .env file
        env_file = "src/presentation/api/.env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    This is the recommended way to access settings throughout the application.

    Returns:
        Settings instance
    """
    return Settings()
