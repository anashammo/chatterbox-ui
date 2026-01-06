#!/usr/bin/env python3
"""
TTS Performance Testing Script

Tests Chatterbox TTS synthesis with Arabic text using the multilingual model.
Measures processing time under different optimization configurations.
"""

import requests
import time
import json
import sys
from typing import Dict, Any, List

API_BASE = "http://localhost:8002/api/v1"

# Arabic test texts of varying lengths
ARABIC_TEXTS = {
    "short": "مرحباً بكم في عالم التكنولوجيا",  # ~30 chars
    "medium": "مرحباً بكم في عالم التكنولوجيا. نحن نقدم لكم أفضل الحلول التقنية لتطوير أعمالكم وتحقيق أهدافكم.",  # ~100 chars
    "long": """عَنْ تَوَكَّلْنَا
هُوَ التَّطْبِيقُ الْوَطَنِيُّ الشَّامِلُ الَّذِي يُوَحِّدُ خَدَمَاتِ الْجِهَاتِ الْحُكُومِيَّةِ فِي مَكَانٍ وَاحِدٍ، حَيْثُ يَجْمَعُ الْخَدَمَاتِ وَالْمَعْلُومَاتِ وَالْوَثَائِقَ وَالْمَنْشُورَاتِ، مِمَّا يُسَهِّلُ الْوُصُولَ إِلَيْهَا وَاسْتِخْدَامَهَا بِكَفَاءَةٍ، وَيُعَزِّزُ جَوْدَةَ الْحَيَاةِ تَمَاشِيًا مَعَ مُسْتَهْدَفَاتِ رُؤْيَةِ الْمَمْلَكَةِ أَلْفَيْنِ وَثَلَاثِينَ.""",  # Tawakkalna description
}


def check_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def get_models() -> List[Dict]:
    """Get available models."""
    try:
        response = requests.get(f"{API_BASE}/models/available", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
        return []
    except Exception as e:
        print(f"Failed to get models: {e}")
        return []


def synthesize(text: str, model: str = "multilingual", language: str = "ar") -> Dict[str, Any]:
    """
    Run synthesis and return timing info.

    Returns dict with:
        - success: bool
        - processing_time: float (from API)
        - request_time: float (wall clock)
        - audio_duration: float
        - error: str (if failed)
    """
    result = {
        "success": False,
        "processing_time": 0.0,
        "request_time": 0.0,
        "audio_duration": 0.0,
        "error": None
    }

    payload = {
        "text": text,
        "model": model,
        "language": language,
        "cfg_weight": 0.5,
        "exaggeration": 0.5
    }

    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE}/syntheses",
            json=payload,
            timeout=300  # 5 min timeout for long synthesis
        )
        result["request_time"] = time.time() - start_time

        if response.status_code in (200, 201):
            data = response.json()
            result["success"] = True
            result["processing_time"] = data.get("processing_time_seconds", 0.0)
            result["audio_duration"] = data.get("output_duration_seconds", 0.0)
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        result["request_time"] = time.time() - start_time
        result["error"] = str(e)

    return result


def run_test(test_name: str, text_key: str, num_runs: int = 3) -> Dict[str, Any]:
    """
    Run multiple synthesis tests and calculate averages.
    """
    text = ARABIC_TEXTS[text_key]
    results = []

    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"Text: {text[:50]}... ({len(text)} chars)")
    print(f"Runs: {num_runs}")
    print(f"{'='*60}")

    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end=" ", flush=True)
        result = synthesize(text, model="multilingual", language="ar")
        results.append(result)

        if result["success"]:
            print(f"OK - Processing: {result['processing_time']:.2f}s, "
                  f"Audio: {result['audio_duration']:.2f}s")
        else:
            print(f"FAILED - {result['error'][:50]}")

    # Calculate stats
    successful = [r for r in results if r["success"]]

    if successful:
        avg_processing = sum(r["processing_time"] for r in successful) / len(successful)
        avg_audio = sum(r["audio_duration"] for r in successful) / len(successful)
        avg_request = sum(r["request_time"] for r in successful) / len(successful)

        # Real-time factor (RTF) = processing_time / audio_duration
        rtf = avg_processing / avg_audio if avg_audio > 0 else 0

        print(f"\nResults ({len(successful)}/{num_runs} successful):")
        print(f"  Avg Processing Time: {avg_processing:.3f}s")
        print(f"  Avg Audio Duration:  {avg_audio:.3f}s")
        print(f"  Avg Request Time:    {avg_request:.3f}s")
        print(f"  Real-Time Factor:    {rtf:.2f}x")

        return {
            "test_name": test_name,
            "text_length": len(text),
            "num_runs": num_runs,
            "successful_runs": len(successful),
            "avg_processing_time": avg_processing,
            "avg_audio_duration": avg_audio,
            "avg_request_time": avg_request,
            "rtf": rtf
        }
    else:
        print(f"\nAll runs failed!")
        return {
            "test_name": test_name,
            "text_length": len(text),
            "num_runs": num_runs,
            "successful_runs": 0,
            "error": results[0]["error"] if results else "Unknown error"
        }


def main():
    print("="*60)
    print("Chatterbox TTS Performance Test")
    print("Model: multilingual, Language: Arabic (ar)")
    print("="*60)

    # Health check
    print("\nChecking API health...", end=" ")
    if not check_health():
        print("FAILED - API not available")
        sys.exit(1)
    print("OK")

    # Check models
    print("\nChecking available models...", end=" ")
    models = get_models()
    multilingual = next((m for m in models if m.get("name") == "multilingual"), None)
    if multilingual:
        print(f"OK - multilingual model {'LOADED' if multilingual.get('is_loaded') else 'not loaded'}")
    else:
        print("WARNING - multilingual model not found")

    # Warmup run (first run loads the model)
    print("\n" + "="*60)
    print("WARMUP RUN (model loading)")
    print("="*60)
    warmup = synthesize(ARABIC_TEXTS["short"], model="multilingual", language="ar")
    if warmup["success"]:
        print(f"Warmup completed in {warmup['request_time']:.2f}s "
              f"(processing: {warmup['processing_time']:.2f}s)")
    else:
        print(f"Warmup failed: {warmup['error']}")
        print("Continuing anyway...")

    # Run performance tests
    all_results = []

    # Test with different text lengths
    for text_key in ["short", "medium", "long"]:
        result = run_test(
            test_name=f"Arabic {text_key.upper()} text",
            text_key=text_key,
            num_runs=3
        )
        all_results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Test Name':<25} {'Chars':<8} {'Proc(s)':<10} {'RTF':<8}")
    print("-"*60)

    for r in all_results:
        if r.get("successful_runs", 0) > 0:
            print(f"{r['test_name']:<25} {r['text_length']:<8} "
                  f"{r['avg_processing_time']:<10.3f} {r['rtf']:<8.2f}x")
        else:
            print(f"{r['test_name']:<25} FAILED")

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

    return all_results


if __name__ == "__main__":
    results = main()
