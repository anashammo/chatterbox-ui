"""
Run the Whisper backend server in development mode.

This script starts the FastAPI backend server using uvicorn with auto-reload
enabled. It loads configuration from environment variables (.env file) and
displays startup information.

Difference from run_backend.py:
    - run_dev.py: Uses settings from .env file, respects all environment variables
    - run_backend.py: Hardcoded to port 8001, simpler startup script

Configuration:
    Settings are loaded from .env file via Settings class:
    - API_HOST (default: 0.0.0.0)
    - API_PORT (default: 8001)
    - WHISPER_MODEL (default: base)
    - WHISPER_DEVICE (default: cuda)
    - LOG_LEVEL (default: info)

Usage:
    python scripts/run_dev.py

Features:
    - Auto-reload on code changes (development mode)
    - Displays startup information (model, device, ports)
    - Loads configuration from .env file
    - Access API docs at http://localhost:{port}/docs

Examples:
    # Start server with default settings (port 8001)
    python scripts/run_dev.py

    # Configure via .env file:
    # API_PORT=8080
    # WHISPER_MODEL=medium
    # WHISPER_DEVICE=cpu
    # Then run:
    python scripts/run_dev.py

Exit:
    Press CTRL+C to stop the server

Note:
    Ensure database is initialized first:
    python scripts/init_db.py
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
    print(f"Whisper Model: {settings.whisper_model}")
    print(f"Device: {settings.whisper_device}")
    print(f"API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print("\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "src.presentation.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
