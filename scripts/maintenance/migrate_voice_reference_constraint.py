"""
Migrate voice_references table constraint from name-only to name+language.

This script updates the PostgreSQL database to allow the same voice reference
name with different languages (e.g., "Anas" in "en" and "Anas" in "ar").

Usage:
    # Run from project root:
    docker exec chatterbox-backend python scripts/maintenance/migrate_voice_reference_constraint.py

    # Or run locally (requires DATABASE_URL environment variable):
    python scripts/maintenance/migrate_voice_reference_constraint.py

Changes:
    - Drops: voice_references_name_key (unique constraint on name only)
    - Adds: uq_voice_reference_name_language (composite unique on name + language)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def migrate_postgresql():
    """Migrate PostgreSQL database constraint."""
    from sqlalchemy import create_engine, text

    # Get database URL from environment
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        print("For Docker: DATABASE_URL=postgresql://chatterbox:password@postgres:5432/chatterbox_tts")
        sys.exit(1)

    print(f"Connecting to database...")
    engine = create_engine(database_url)

    with engine.connect() as conn:
        # Check if old constraint exists
        print("\nChecking existing constraints...")
        result = conn.execute(text("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'voice_references'
            AND constraint_type = 'UNIQUE'
        """))
        constraints = [row[0] for row in result]
        print(f"Found constraints: {constraints}")

        # Drop old name-only constraint if exists
        if 'voice_references_name_key' in constraints:
            print("\nDropping old constraint: voice_references_name_key")
            conn.execute(text(
                "ALTER TABLE voice_references DROP CONSTRAINT voice_references_name_key"
            ))
            conn.commit()
            print("Old constraint dropped successfully")
        else:
            print("\nOld constraint 'voice_references_name_key' not found (already migrated?)")

        # Check if new constraint already exists
        result = conn.execute(text("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'voice_references'
            AND constraint_type = 'UNIQUE'
        """))
        constraints = [row[0] for row in result]

        if 'uq_voice_reference_name_language' in constraints:
            print("\nNew constraint 'uq_voice_reference_name_language' already exists")
        else:
            print("\nAdding new constraint: uq_voice_reference_name_language")
            conn.execute(text("""
                ALTER TABLE voice_references
                ADD CONSTRAINT uq_voice_reference_name_language
                UNIQUE (name, language)
            """))
            conn.commit()
            print("New constraint added successfully")

        # Verify final state
        print("\nVerifying final constraints...")
        result = conn.execute(text("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'voice_references'
            AND constraint_type = 'UNIQUE'
        """))
        constraints = [row[0] for row in result]
        print(f"Final constraints: {constraints}")

        # Show table structure
        print("\nTable structure:")
        result = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'voice_references'
            ORDER BY ordinal_position
        """))
        for row in result:
            print(f"  {row[0]}: {row[1]} {'(nullable)' if row[2] == 'YES' else '(not null)'}")

    print("\nMigration completed successfully!")


def main():
    print("=" * 60)
    print("Voice Reference Constraint Migration")
    print("=" * 60)
    print("\nThis script migrates the unique constraint from:")
    print("  - name (unique)")
    print("To:")
    print("  - name + language (composite unique)")
    print("\nThis allows the same name with different languages.")
    print()

    migrate_postgresql()


if __name__ == "__main__":
    main()
