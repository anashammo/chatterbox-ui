"""Check Chatterbox TTS database status"""
import sys
from pathlib import Path

# Add project root to path (scripts/maintenance/ -> scripts/ -> project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.infrastructure.persistence.database import SessionLocal
from src.infrastructure.persistence.models.synthesis_model import SynthesisModel
from src.infrastructure.persistence.models.voice_reference_model import VoiceReferenceModel

db = SessionLocal()
try:
    # Get all voice references
    voice_refs = db.query(VoiceReferenceModel).all()
    print(f'Total voice references: {len(voice_refs)}')
    print()

    for vr in voice_refs[:5]:  # Show first 5
        print(f'Voice Reference ID: {vr.id[:16]}...')
        print(f'  Name: {vr.name}')
        print(f'  Duration: {vr.duration_seconds}s')
        print(f'  Created: {vr.created_at}')
        print()

    # Get all syntheses
    syntheses = db.query(SynthesisModel).all()
    print(f'Total syntheses: {len(syntheses)}')
    print()

    for s in syntheses[:5]:  # Show first 5
        print(f'Synthesis ID: {s.id[:16]}...')
        text = s.input_text or ""
        print(f'  Text: {text[:50]}...' if len(text) > 50 else f'  Text: {text}')
        print(f'  Model: {s.model.value if s.model else "unknown"} | Status: {s.status.value if s.status else "unknown"}')
        if s.output_duration_seconds:
            print(f'  Audio Duration: {s.output_duration_seconds:.2f}s')
        if s.processing_time_seconds:
            print(f'  Processing Time: {s.processing_time_seconds:.2f}s')
        if s.voice_reference_id:
            print(f'  Voice Reference: {s.voice_reference_id[:16]}...')
        print()
finally:
    db.close()
