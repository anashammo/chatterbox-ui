"""
Run the Chatterbox TTS backend server in development mode.

This script starts the FastAPI backend server using uvicorn with auto-reload
enabled. It loads configuration from environment variables and displays
startup information.

Difference from run_backend.py:
    - run_dev.py: Uses settings from .env file, respects all environment variables
    - run_backend.py: Hardcoded to port 8002, simpler startup script

Configuration:
    Settings are loaded from src/presentation/api/.env via Settings class:
    - API_HOST (default: 0.0.0.0)
    - API_PORT (default: 8002)
    - TTS_DEFAULT_MODEL (default: turbo)
    - TTS_DEVICE (default: cuda)
    - LOG_LEVEL (default: INFO)

Usage:
    python scripts/server/run_dev.py

Features:
    - Auto-reload on code changes (development mode)
    - Displays startup information (model, device, ports)
    - Loads configuration from src/presentation/api/.env
    - Access API docs at http://localhost:{port}/docs

Examples:
    # Start server with default settings (port 8002)
    python scripts/server/run_dev.py

    # Configure via src/presentation/api/.env:
    # API_PORT=8080
    # TTS_DEFAULT_MODEL=standard
    # TTS_DEVICE=cpu
    # Then run:
    python scripts/server/run_dev.py

Exit:
    Press CTRL+C to stop the server

Note:
    Ensure database is initialized first:
    python scripts/setup/init_db.py
"""
import uvicorn
import sys
from pathlib import Path

# Add project root to path (scripts/server/ -> scripts/ -> project root)
# Path(__file__).parent = scripts/server/, .parent.parent = scripts/, .parent.parent.parent = project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.infrastructure.config.settings import get_settings


if __name__ == "__main__":
    settings = get_settings()

    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Server: {settings.api_host}:{settings.api_port}")
    print(f"TTS Model: {settings.tts_default_model}")
    print(f"Device: {settings.tts_device}")
    print(f"API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print("\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "src.presentation.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
