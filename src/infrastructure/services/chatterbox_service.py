"""
Chatterbox TTS service implementation.

This service wraps the Resemble AI Chatterbox library for text-to-speech synthesis.

IMPORTANT: The GPU inference is CPU-bound and blocks the event loop. To prevent
blocking all other API requests during synthesis, we run the inference in a
thread pool executor using asyncio.to_thread().
"""

import os
# Fix for transformers SDPA attention issue with Chatterbox
# Must be set BEFORE importing transformers/chatterbox
# See: https://github.com/resemble-ai/chatterbox/issues/339
# SDPA can be enabled via ENABLE_SDPA_ATTENTION=true env var if PR #398 is merged
if os.environ.get('ENABLE_SDPA_ATTENTION', 'false').lower() != 'true':
    os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'

import io
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch
import soundfile as sf
import numpy as np

from ...domain.services.text_to_speech_service import TextToSpeechService
from ...domain.value_objects.tts_model_info import (
    TTSModelInfo,
    SUPPORTED_LANGUAGES,
    is_supported_language,
)
from ...domain.exceptions import ServiceException
from ..config.settings import Settings
from .performance_monitor import PerformanceMonitor
from .text_chunker import chunk_text
from .audio_utils import process_audio

logger = logging.getLogger(__name__)


class ChatterboxService(TextToSpeechService):
    """
    Chatterbox TTS implementation using Resemble AI's Chatterbox library.

    Supports:
    - Multiple TTS models (turbo, standard, multilingual)
    - Zero-shot voice cloning from ~10s audio reference
    - Paralinguistic tags ([laugh], [cough], [sigh], etc.)
    - CFG weight and exaggeration tuning for standard model
    - 23+ languages for multilingual model

    The service lazily loads models on first use to optimize startup time.
    Models are cached in GPU memory for fast subsequent synthesis.
    """

    def __init__(self, settings: Settings):
        """
        Initialize Chatterbox service.

        Args:
            settings: Application settings containing TTS configuration.
        """
        self.settings = settings
        self.device = settings.tts_device

        # Lazy-loaded model instances
        self._turbo_model = None
        self._standard_model = None
        self._multilingual_model = None

        # Track which models are loaded
        self._loaded_models: Dict[str, bool] = {
            "turbo": False,
            "standard": False,
            "multilingual": False,
        }

        # Thread pool for running blocking GPU inference
        # Use a single worker to prevent GPU memory issues from concurrent inference
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts_inference")

        # Voice reference tensor cache: path -> (mtime, tensor)
        # Caches loaded voice references to avoid repeated disk I/O
        self._voice_cache: Dict[str, Tuple[float, torch.Tensor]] = {}
        self._voice_cache_max_size = getattr(settings, 'voice_cache_max_size', 10)
        self._enable_voice_caching = getattr(settings, 'enable_voice_caching', True)

        # Verify CUDA availability and log GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.warning("CUDA not available - using CPU (will be slow)")

        logger.info(f"ChatterboxService initialized with device: {self.device}")

    def _get_turbo_model(self):
        """Lazy load the Turbo model with optional torch.compile() optimization."""
        if self._turbo_model is None:
            logger.info("Loading Chatterbox Turbo model...")
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self._turbo_model = ChatterboxTurboTTS.from_pretrained(device=self.device)

                # Apply torch.compile() for JIT optimization if enabled
                if getattr(self.settings, 'enable_torch_compile', False):
                    logger.info("Compiling Turbo model with torch.compile()...")
                    try:
                        self._turbo_model = torch.compile(
                            self._turbo_model,
                            mode="reduce-overhead"
                        )
                        logger.info("Turbo model compiled successfully")
                    except Exception as compile_error:
                        logger.warning(f"torch.compile() failed, using uncompiled model: {compile_error}")

                self._loaded_models["turbo"] = True
                logger.info("Chatterbox Turbo model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Turbo model: {e}")
                raise ServiceException(f"Failed to load Turbo model: {str(e)}")
        return self._turbo_model

    def _get_standard_model(self):
        """Lazy load the Standard model with optional torch.compile() optimization."""
        if self._standard_model is None:
            logger.info("Loading Chatterbox Standard model...")
            try:
                from chatterbox.tts import ChatterboxTTS
                self._standard_model = ChatterboxTTS.from_pretrained(device=self.device)

                # Apply torch.compile() for JIT optimization if enabled
                if getattr(self.settings, 'enable_torch_compile', False):
                    logger.info("Compiling Standard model with torch.compile()...")
                    try:
                        self._standard_model = torch.compile(
                            self._standard_model,
                            mode="reduce-overhead"
                        )
                        logger.info("Standard model compiled successfully")
                    except Exception as compile_error:
                        logger.warning(f"torch.compile() failed, using uncompiled model: {compile_error}")

                self._loaded_models["standard"] = True
                logger.info("Chatterbox Standard model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Standard model: {e}")
                raise ServiceException(f"Failed to load Standard model: {str(e)}")
        return self._standard_model

    def _get_multilingual_model(self):
        """Lazy load the Multilingual model with optional torch.compile() optimization."""
        if self._multilingual_model is None:
            logger.info("Loading Chatterbox Multilingual model...")
            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self._multilingual_model = ChatterboxMultilingualTTS.from_pretrained(
                    device=self.device
                )

                # Apply torch.compile() for JIT optimization if enabled
                if getattr(self.settings, 'enable_torch_compile', False):
                    logger.info("Compiling Multilingual model with torch.compile()...")
                    try:
                        self._multilingual_model = torch.compile(
                            self._multilingual_model,
                            mode="reduce-overhead"
                        )
                        logger.info("Multilingual model compiled successfully")
                    except Exception as compile_error:
                        logger.warning(f"torch.compile() failed, using uncompiled model: {compile_error}")

                self._loaded_models["multilingual"] = True
                logger.info("Chatterbox Multilingual model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Multilingual model: {e}")
                raise ServiceException(f"Failed to load Multilingual model: {str(e)}")
        return self._multilingual_model

    def _load_voice_reference(self, voice_reference_path: str) -> torch.Tensor:
        """
        Load and preprocess voice reference audio.

        Args:
            voice_reference_path: Path to the voice reference audio file.

        Returns:
            Audio tensor ready for voice cloning.
        """
        try:
            audio, sr = torchaudio.load(voice_reference_path)
            # Resample to model's expected sample rate if needed (24kHz)
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                audio = resampler(audio)
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            return audio.squeeze(0)
        except Exception as e:
            raise ServiceException(f"Failed to load voice reference: {str(e)}")

    def _get_cached_voice_reference(self, path: str) -> torch.Tensor:
        """
        Load voice reference with caching based on file modification time.

        Caches loaded voice reference tensors to avoid repeated disk I/O.
        Cache invalidation is based on file mtime - if file changes, cache is refreshed.
        Uses LRU-style eviction when cache is full.

        Args:
            path: Path to the voice reference audio file.

        Returns:
            Audio tensor ready for voice cloning.
        """
        if not self._enable_voice_caching:
            return self._load_voice_reference(path)

        mtime = os.path.getmtime(path)

        # Check cache
        if path in self._voice_cache:
            cached_mtime, cached_tensor = self._voice_cache[path]
            if cached_mtime == mtime:
                logger.debug(f"Voice reference cache hit: {path}")
                return cached_tensor

        # Load fresh
        logger.debug(f"Voice reference cache miss: {path}")
        tensor = self._load_voice_reference(path)

        # Evict oldest if cache full (LRU-style: remove first inserted)
        if len(self._voice_cache) >= self._voice_cache_max_size:
            oldest_key = next(iter(self._voice_cache))
            del self._voice_cache[oldest_key]
            logger.debug(f"Evicted from voice cache: {oldest_key}")

        self._voice_cache[path] = (mtime, tensor)
        return tensor

    def clear_voice_cache(self) -> None:
        """Clear the voice reference cache to free memory."""
        self._voice_cache.clear()
        logger.info("Voice reference cache cleared")

    def _run_inference_sync(
        self,
        text: str,
        model: str,
        voice_reference_path: Optional[str],
        language: Optional[str],
        cfg_weight: float,
        exaggeration: float,
    ) -> torch.Tensor:
        """
        Run the blocking TTS inference synchronously.

        This method is meant to be run in a thread pool to avoid blocking
        the async event loop.

        Uses torch.inference_mode() for optimal inference performance
        (more efficient than torch.no_grad() as it also disables version counting).

        CUDA synchronization is used before and after inference to ensure
        accurate timing measurements when using GPU acceleration.

        Args:
            text: The text to convert to speech.
            model: TTS model to use.
            voice_reference_path: Path to voice reference audio.
            language: Language code for multilingual model.
            cfg_weight: Configuration weight.
            exaggeration: Speech expressiveness.

        Returns:
            Audio waveform tensor.
        """
        # CUDA sync before inference for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.inference_mode():
            if model == "turbo":
                tts_model = self._get_turbo_model()
                wav = tts_model.generate(
                    text=text,
                    audio_prompt_path=voice_reference_path,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                )
            elif model == "standard":
                tts_model = self._get_standard_model()
                wav = tts_model.generate(
                    text=text,
                    audio_prompt_path=voice_reference_path,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                )
            else:  # multilingual
                tts_model = self._get_multilingual_model()
                wav = tts_model.generate(
                    text=text,
                    language_id=language,
                    audio_prompt_path=voice_reference_path,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                )

            # CUDA sync after inference for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            return wav

    async def synthesize(
        self,
        text: str,
        model: str = "turbo",
        voice_reference_path: Optional[str] = None,
        language: Optional[str] = None,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text using Chatterbox TTS.

        The GPU inference runs in a thread pool to avoid blocking the async
        event loop, allowing other API requests to be served concurrently.

        Args:
            text: The text to convert to speech.
            model: TTS model to use (turbo, standard, multilingual).
            voice_reference_path: Path to voice reference audio for cloning.
            language: Language code for multilingual model.
            cfg_weight: Configuration weight for accent transfer (0-1).
            exaggeration: Speech expressiveness (0-1+).

        Returns:
            Dict containing:
                - audio_data: bytes - Raw WAV audio data
                - duration_seconds: float - Audio duration
                - sample_rate: int - Audio sample rate (24000)

        Raises:
            ValueError: If parameters are invalid.
            ServiceException: If synthesis fails.
        """
        # Initialize performance monitoring
        enable_perf_logging = getattr(self.settings, 'enable_performance_logging', True)
        monitor = PerformanceMonitor(name=f"tts_{model}") if enable_perf_logging else None

        # Validate parameters
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if model not in ["turbo", "standard", "multilingual"]:
            raise ValueError(f"Invalid model: {model}. Must be turbo, standard, or multilingual")

        if model == "multilingual" and not language:
            raise ValueError("Language is required for multilingual model")

        if language and not is_supported_language(language):
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )

        if monitor:
            monitor.record("validation")

        try:
            # Run blocking GPU inference in thread pool to avoid blocking event loop
            # This allows other API requests to be served while synthesis is running
            loop = asyncio.get_event_loop()
            wav = await loop.run_in_executor(
                self._executor,
                self._run_inference_sync,
                text,
                model,
                voice_reference_path,
                language,
                cfg_weight,
                exaggeration,
            )

            if monitor:
                monitor.record("inference")

            # Calculate duration
            sample_rate = 24000  # Chatterbox outputs at 24kHz

            # Convert to numpy for soundfile
            if isinstance(wav, torch.Tensor):
                audio_np = wav.cpu().numpy()
            else:
                audio_np = np.array(wav)

            # Ensure 1D array
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()

            if monitor:
                monitor.record("tensor_convert")

            # Optional audio post-processing (silence trimming, fade)
            if getattr(self.settings, 'audio_trim_silence', True):
                audio_np = process_audio(
                    audio_np,
                    sample_rate=sample_rate,
                    trim=True,
                    normalize=False,
                    fade=True,
                )
                if monitor:
                    monitor.record("post_process")

            duration_seconds = len(audio_np) / sample_rate

            # Convert to bytes (WAV format) using soundfile
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_np, sample_rate, format='WAV')
            audio_data = audio_buffer.getvalue()

            if monitor:
                monitor.record("encoding")

            # Get processing time from monitor or calculate manually
            processing_time = monitor.report()["total"] if monitor else 0.0

            # Log performance details
            if monitor:
                monitor.log_report(level="info")
            else:
                logger.info(
                    f"Synthesis completed: model={model}, "
                    f"duration={duration_seconds:.2f}s"
                )

            # Optional GPU memory cleanup
            if getattr(self.settings, 'enable_gpu_memory_cleanup', False):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return {
                "audio_data": audio_data,
                "duration_seconds": duration_seconds,
                "sample_rate": sample_rate,
                "processing_time_seconds": processing_time,
            }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise ServiceException(f"TTS synthesis failed: {str(e)}")

    async def synthesize_long_text(
        self,
        text: str,
        model: str = "turbo",
        voice_reference_path: Optional[str] = None,
        language: Optional[str] = None,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
        max_chunk_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize long text by chunking into sentences.

        Maintains quality for long text by processing in smaller chunks
        and concatenating the audio. Uses settings-based chunk size by default.

        Args:
            text: Text to synthesize (can be long)
            model: TTS model to use
            voice_reference_path: Path to voice reference audio
            language: Language code for multilingual model
            cfg_weight: CFG weight parameter
            exaggeration: Exaggeration parameter
            max_chunk_chars: Maximum characters per chunk (uses settings default if None)

        Returns:
            Dict with combined audio_data, duration_seconds, processing_time_seconds, chunks_processed
        """
        # Check if text chunking is enabled
        if not getattr(self.settings, 'enable_text_chunking', True):
            return await self.synthesize(
                text, model, voice_reference_path, language, cfg_weight, exaggeration
            )

        # Use settings-based chunk size if not specified
        if max_chunk_chars is None:
            max_chunk_chars = getattr(self.settings, 'tts_chunk_max_chars', 250)

        chunks = chunk_text(text, max_chars=max_chunk_chars)

        if len(chunks) <= 1:
            # Single chunk, use regular synthesis
            return await self.synthesize(
                text, model, voice_reference_path, language, cfg_weight, exaggeration
            )

        logger.info(f"Processing {len(chunks)} chunks for long text synthesis")

        audio_segments = []
        total_duration = 0.0
        total_processing_time = 0.0
        sample_rate = 24000

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}: {len(chunk)} chars")

            result = await self.synthesize(
                text=chunk,
                model=model,
                voice_reference_path=voice_reference_path,
                language=language,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
            )

            # Read audio data back as numpy
            audio_buffer = io.BytesIO(result["audio_data"])
            audio_np, _ = sf.read(audio_buffer)
            audio_segments.append(audio_np)

            total_duration += result["duration_seconds"]
            total_processing_time += result["processing_time_seconds"]

        # Concatenate with small silence between chunks
        silence_samples = int(0.1 * sample_rate)  # 100ms silence
        silence = np.zeros(silence_samples, dtype=np.float32)

        combined_parts = []
        for i, segment in enumerate(audio_segments):
            combined_parts.append(segment)
            if i < len(audio_segments) - 1:
                combined_parts.append(silence)

        combined_audio = np.concatenate(combined_parts)

        # Encode combined audio
        output_buffer = io.BytesIO()
        sf.write(output_buffer, combined_audio, sample_rate, format='WAV')

        return {
            "audio_data": output_buffer.getvalue(),
            "duration_seconds": total_duration + (len(chunks) - 1) * 0.1,
            "sample_rate": sample_rate,
            "processing_time_seconds": total_processing_time,
            "chunks_processed": len(chunks),
        }

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available TTS models with metadata.

        Returns:
            List of dicts with model information.
        """
        models = TTSModelInfo.get_all_models()
        return [
            {
                "name": m.name,
                "display_name": m.display_name,
                "parameters": m.parameters,
                "description": m.description,
                "supports_voice_cloning": m.supports_voice_cloning,
                "supports_multilingual": m.supports_multilingual,
                "supports_paralinguistics": m.supports_paralinguistics,
                "is_loaded": self._loaded_models.get(m.name, False),
            }
            for m in models
        ]

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes for multilingual model.

        Returns:
            List of ISO language codes.
        """
        return list(SUPPORTED_LANGUAGES)

    def is_model_loaded(self, model: str) -> bool:
        """
        Check if a specific model is currently loaded in memory.

        Args:
            model: Model name to check.

        Returns:
            True if model is loaded and ready for inference.
        """
        return self._loaded_models.get(model, False)

    def preload_model(self, model: str) -> None:
        """
        Preload a model into memory.

        Args:
            model: Model name to preload (turbo, standard, multilingual).

        Raises:
            ValueError: If model name is invalid.
        """
        if model == "turbo":
            self._get_turbo_model()
        elif model == "standard":
            self._get_standard_model()
        elif model == "multilingual":
            self._get_multilingual_model()
        else:
            raise ValueError(f"Invalid model: {model}")

    def unload_model(self, model: str) -> None:
        """
        Unload a model from memory to free GPU resources.

        Args:
            model: Model name to unload.
        """
        if model == "turbo" and self._turbo_model is not None:
            del self._turbo_model
            self._turbo_model = None
            self._loaded_models["turbo"] = False
            logger.info("Turbo model unloaded")
        elif model == "standard" and self._standard_model is not None:
            del self._standard_model
            self._standard_model = None
            self._loaded_models["standard"] = False
            logger.info("Standard model unloaded")
        elif model == "multilingual" and self._multilingual_model is not None:
            del self._multilingual_model
            self._multilingual_model = None
            self._loaded_models["multilingual"] = False
            logger.info("Multilingual model unloaded")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information for diagnostics.

        Returns:
            Dict with GPU info (name, memory, CUDA version).
        """
        if not torch.cuda.is_available():
            return {"available": False, "device": "cpu"}

        return {
            "available": True,
            "device": self.device,
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
            "cuda_version": torch.version.cuda,
        }
