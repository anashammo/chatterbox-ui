"""Audio post-processing utilities."""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 24000,
    top_db: float = 30.0,
) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.

    Args:
        audio: Audio waveform as numpy array
        sample_rate: Audio sample rate
        top_db: Threshold in dB below peak to consider as silence

    Returns:
        Trimmed audio array
    """
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)

        original_duration = len(audio) / sample_rate
        trimmed_duration = len(trimmed) / sample_rate

        if original_duration - trimmed_duration > 0.1:
            logger.debug(
                f"Trimmed {original_duration - trimmed_duration:.2f}s silence "
                f"({original_duration:.2f}s -> {trimmed_duration:.2f}s)"
            )

        return trimmed
    except ImportError:
        logger.warning("librosa not available, skipping silence trimming")
        return audio
    except Exception as e:
        logger.warning(f"Silence trimming failed: {e}")
        return audio


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -3.0,
) -> np.ndarray:
    """
    Normalize audio to target dB level.

    Args:
        audio: Audio waveform as numpy array
        target_db: Target peak level in dB (default -3 dB)

    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio

    # Calculate current peak
    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    # Calculate target peak from dB
    target_peak = 10 ** (target_db / 20)

    # Apply gain
    gain = target_peak / peak
    normalized = audio * gain

    return normalized


def apply_fade(
    audio: np.ndarray,
    sample_rate: int = 24000,
    fade_in_ms: float = 10.0,
    fade_out_ms: float = 10.0,
) -> np.ndarray:
    """
    Apply fade in/out to audio to prevent clicks.

    Args:
        audio: Audio waveform as numpy array
        sample_rate: Audio sample rate
        fade_in_ms: Fade in duration in milliseconds
        fade_out_ms: Fade out duration in milliseconds

    Returns:
        Audio with fades applied
    """
    audio = audio.copy()

    fade_in_samples = int(fade_in_ms * sample_rate / 1000)
    fade_out_samples = int(fade_out_ms * sample_rate / 1000)

    # Fade in
    if fade_in_samples > 0 and len(audio) > fade_in_samples:
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in_curve

    # Fade out
    if fade_out_samples > 0 and len(audio) > fade_out_samples:
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out_curve

    return audio


def process_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    trim: bool = True,
    normalize: bool = False,
    fade: bool = True,
    trim_db: float = 30.0,
    normalize_db: float = -3.0,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """
    Apply full audio post-processing pipeline.

    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        trim: Whether to trim silence
        normalize: Whether to normalize volume
        fade: Whether to apply fade in/out
        trim_db: Silence threshold for trimming
        normalize_db: Target level for normalization
        fade_ms: Fade duration in milliseconds

    Returns:
        Processed audio array
    """
    result = audio

    if trim:
        result = trim_silence(result, sample_rate, trim_db)

    if normalize:
        result = normalize_audio(result, normalize_db)

    if fade:
        result = apply_fade(result, sample_rate, fade_ms, fade_ms)

    return result
