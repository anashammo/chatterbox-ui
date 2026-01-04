"""Show Chatterbox TTS database contents"""
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

    if voice_refs:
        print('\nVoice References:')
        for vr in voice_refs:
            print(f'  - ID: {vr.id[:16]}...')
            print(f'    Name: {vr.name}')
            print(f'    Duration: {vr.duration_seconds}s')
            print(f'    Created: {vr.created_at}')
            print()
    else:
        print('  (No voice references in database)')

    # Get all syntheses
    syntheses = db.query(SynthesisModel).all()
    print(f'\nTotal syntheses: {len(syntheses)}')

    if syntheses:
        print('\nSyntheses:')
        for s in syntheses:
            print(f'  - ID: {s.id[:16]}...')
            text = s.input_text or ""
            print(f'    Text: {text[:40]}...' if len(text) > 40 else f'    Text: {text}')
            print(f'    Model: {s.model.value if s.model else "unknown"} | Status: {s.status.value if s.status else "unknown"}')
            if s.voice_reference_id:
                print(f'    Voice Ref: {s.voice_reference_id[:16]}...')
            print()
    else:
        print('  (No syntheses in database)')

finally:
    db.close()
