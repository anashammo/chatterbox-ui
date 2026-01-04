"""
Initialize the Chatterbox TTS database.

This script creates the SQLite database with all required tables for the
Chatterbox TTS system. It is idempotent - safe to run multiple times
without deleting existing data.

Tables Created:
    - syntheses: Stores synthesis results with status tracking
      - id, text, model, voice_reference_id (FK), status, language,
        audio_file_path, duration_seconds, cfg_weight, exaggeration,
        created_at, completed_at, error_message, processing_time_seconds

    - voice_references: Stores voice reference metadata for cloning
      - id, name, description, file_path, duration_seconds, created_at

Database Location:
    - Default: ./chatterbox_tts.db (project root)
    - Configurable via DATABASE_URL environment variable

Usage:
    python scripts/setup/init_db.py

Exit Codes:
    0: Success - Database initialized or already exists
    1: Failure - Error during initialization

Examples:
    # Initialize database with default settings
    python scripts/setup/init_db.py

    # Database will be created if it doesn't exist
    # Existing data will NOT be deleted
"""
import sys
from pathlib import Path

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path (scripts/setup/ -> scripts/ -> project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.infrastructure.persistence.database import init_db


if __name__ == "__main__":
    print("Initializing Chatterbox TTS database...")
    try:
        init_db()
        print("Database initialized successfully!")
        print("All tables created")
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)
