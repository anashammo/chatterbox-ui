#!/usr/bin/env python3
"""
Smart pre-download Chatterbox TTS models for Docker container startup.

This script downloads all available TTS models from HuggingFace Hub
before the FastAPI server starts, ensuring faster first inference.

Features (similar to Whisper ASR smart pre-download):
- Checks if models are already cached
- Shows cache file sizes
- Skips already-cached models (use --force to re-download)
- Shows download progress
- Supports selective model download

Models:
- turbo (ChatterboxTurboTTS) - 350M parameters, fastest
- standard (ChatterboxTTS) - 500M parameters, best quality
- multilingual (ChatterboxMultilingualTTS) - 500M parameters, 23+ languages
"""
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# Model repository mappings for HuggingFace Hub
MODEL_REPOS = {
    'turbo': 'resemble-ai/chatterbox',
    'standard': 'resemble-ai/chatterbox',
    'multilingual': 'resemble-ai/chatterbox'
}


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_cache_dir() -> Path:
    """Get the HuggingFace cache directory."""
    cache_home = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    return Path(cache_home) / 'hub'


def get_cached_model_size(model_name: str) -> Tuple[bool, Optional[int]]:
    """
    Check if model is already cached and return its size.

    Returns:
        Tuple of (is_cached, size_in_bytes)
    """
    try:
        from huggingface_hub import scan_cache_dir, HfFolder

        cache_info = scan_cache_dir()

        # Chatterbox models are all in the same repo
        for repo in cache_info.repos:
            if 'chatterbox' in repo.repo_id.lower() or 'resemble' in repo.repo_id.lower():
                # Calculate size of all cached files for this repo
                total_size = sum(
                    rev.size_on_disk for rev in repo.revisions
                )
                if total_size > 0:
                    return True, total_size

        return False, None

    except Exception as e:
        # If we can't check the cache, assume not cached
        return False, None


def download_model(model_name: str, force: bool = False) -> bool:
    """
    Download a specific Chatterbox TTS model.

    Args:
        model_name: Name of model (turbo, standard, multilingual)
        force: If True, re-download even if cached

    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing '{model_name}':")

    # Check cache status
    is_cached, cache_size = get_cached_model_size(model_name)

    if is_cached and cache_size:
        print(f"  Model '{model_name}' found in cache ({format_size(cache_size)})")

        if not force:
            print(f"  Skipping '{model_name}' (already cached, use --force to re-download)")
            return True
        else:
            print(f"  Force flag set, re-downloading '{model_name}'...")
    else:
        print(f"  Model '{model_name}' not found in cache")

    print(f"  Downloading model '{model_name}' from HuggingFace Hub...")
    start_time = time.time()

    try:
        # Import the appropriate model class based on model name
        # Loading the model will download it if not cached
        if model_name == 'turbo':
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            print("  Loading ChatterboxTurboTTS...")
            model = ChatterboxTurboTTS.from_pretrained(device='cpu')
        elif model_name == 'standard':
            from chatterbox.tts import ChatterboxTTS
            print("  Loading ChatterboxTTS...")
            model = ChatterboxTTS.from_pretrained(device='cpu')
        elif model_name == 'multilingual':
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            print("  Loading ChatterboxMultilingualTTS...")
            model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
        else:
            print(f"  Unknown model: {model_name}")
            return False

        elapsed = time.time() - start_time
        print(f"  Successfully downloaded '{model_name}' in {elapsed:.1f}s")

        # Clean up to free memory
        del model

        # Check final cache size
        _, final_size = get_cached_model_size(model_name)
        if final_size:
            print(f"  Cache size: {format_size(final_size)}")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  Error downloading '{model_name}' after {elapsed:.1f}s: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Smart pre-download Chatterbox TTS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  turbo        350M parameters, fastest, single-step generation
  standard     500M parameters, best quality, cfg/exaggeration tuning
  multilingual 500M parameters, 23+ languages, zero-shot voice cloning

Examples:
  %(prog)s                          # Download all models
  %(prog)s --models turbo           # Download only turbo model
  %(prog)s --models turbo standard  # Download turbo and standard
  %(prog)s --force                  # Force re-download all models
"""
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['turbo', 'standard', 'multilingual'],
        help='Models to download (default: all). Supports comma-separated values.'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if cached'
    )

    args = parser.parse_args()

    # Handle comma-separated values (e.g., "turbo,standard,multilingual")
    models = []
    for m in args.models:
        # Split by comma and strip whitespace
        models.extend([x.strip() for x in m.split(',') if x.strip()])

    # Validate model names
    valid_models = {'turbo', 'standard', 'multilingual'}
    invalid = [m for m in models if m not in valid_models]
    if invalid:
        print(f"Error: Invalid model(s): {', '.join(invalid)}")
        print(f"Valid models: {', '.join(valid_models)}")
        sys.exit(1)

    # Remove duplicates while preserving order
    seen = set()
    models = [m for m in models if not (m in seen or seen.add(m))]

    print("=" * 60)
    print("Chatterbox TTS Smart Model Pre-Download")
    print("=" * 60)
    print(f"\nModels to process: {', '.join(models)}")
    print(f"Force re-download: {args.force}")
    print(f"Cache directory: {get_cache_dir()}")

    # Track results
    success_count = 0
    fail_count = 0
    skipped_count = 0

    for model_name in models:
        result = download_model(model_name, args.force)
        if result:
            # Check if it was already cached (skipped)
            is_cached, _ = get_cached_model_size(model_name)
            if is_cached and not args.force:
                skipped_count += 1
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {success_count} successful, {fail_count} failed")
    if skipped_count > 0:
        print(f"         ({skipped_count} already cached, skipped)")
    print("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
