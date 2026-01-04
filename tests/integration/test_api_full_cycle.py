#!/usr/bin/env python3
"""
Full API Integration Test Suite for Chatterbox TTS

Tests all API endpoints with full CRUD cycle coverage.
"""

import requests
import os
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8002/api/v1"
TEST_DIR = Path(__file__).parent
TEST_AUDIO_FILE = TEST_DIR / "test_voice_reference.wav"

# Test results tracking
results = {"passed": 0, "failed": 0, "tests": []}


def log_result(test_name: str, passed: bool, details: str = ""):
    """Log test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    results["passed" if passed else "failed"] += 1
    results["tests"].append({"name": test_name, "passed": passed, "details": details})
    print(f"{status}: {test_name}")
    if details and not passed:
        print(f"       Details: {details}")


def create_test_audio():
    """Create a test audio file for voice reference testing."""
    print("\n=== Creating Test Audio File ===")
    sample_rate = 24000
    duration = 10  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    os.makedirs(TEST_DIR, exist_ok=True)
    sf.write(str(TEST_AUDIO_FILE), audio, sample_rate)
    print(f"Created: {TEST_AUDIO_FILE} ({duration}s @ {sample_rate}Hz)")
    return TEST_AUDIO_FILE


# =============================================================================
# Health Endpoint Tests
# =============================================================================

def test_health_endpoint():
    """Test GET /health"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        passed = response.status_code == 200 and response.json().get("status") == "healthy"
        log_result("GET /health", passed, f"Status: {response.status_code}")
        return passed
    except Exception as e:
        log_result("GET /health", False, str(e))
        return False


# =============================================================================
# Model Endpoint Tests
# =============================================================================

def test_available_models():
    """Test GET /models/available"""
    print("\n=== Testing Model Endpoints ===")
    try:
        response = requests.get(f"{BASE_URL}/models/available", timeout=10)
        data = response.json()
        passed = response.status_code == 200 and "models" in data and len(data["models"]) >= 3
        log_result("GET /models/available", passed, f"Found {len(data.get('models', []))} models")

        # Check each model has required fields
        for model in data.get("models", []):
            has_fields = all(k in model for k in ["name", "display_name", "supports_voice_cloning"])
            log_result(f"  Model '{model.get('name')}' structure", has_fields)

        return passed
    except Exception as e:
        log_result("GET /models/available", False, str(e))
        return False


def test_model_status():
    """Test GET /models/status/{model}"""
    try:
        for model in ["turbo", "standard", "multilingual"]:
            response = requests.get(f"{BASE_URL}/models/status/{model}", timeout=10)
            passed = response.status_code == 200
            log_result(f"GET /models/status/{model}", passed, f"Status: {response.status_code}")
        return True
    except Exception as e:
        log_result("GET /models/status", False, str(e))
        return False


# =============================================================================
# Voice Reference Endpoint Tests
# =============================================================================

def test_voice_reference_crud():
    """Test full CRUD cycle for voice references."""
    print("\n=== Testing Voice Reference CRUD ===")
    voice_ref_id = None

    try:
        # 1. LIST (empty)
        response = requests.get(f"{BASE_URL}/voice-references", timeout=10)
        initial_count = response.json().get("total", 0)
        log_result("GET /voice-references (initial)", response.status_code == 200)

        # 2. CREATE (upload)
        with open(TEST_AUDIO_FILE, "rb") as f:
            files = {"file": ("test_voice.wav", f, "audio/wav")}
            data = {"name": "Test Voice", "description": "Integration test voice reference"}
            response = requests.post(f"{BASE_URL}/voice-references", files=files, data=data, timeout=30)

        if response.status_code == 201 or response.status_code == 200:
            voice_ref_id = response.json().get("id")
            log_result("POST /voice-references (upload)", True, f"ID: {voice_ref_id}")
        else:
            log_result("POST /voice-references (upload)", False, f"Status: {response.status_code}, Body: {response.text[:200]}")
            return None

        # 3. LIST (verify created)
        response = requests.get(f"{BASE_URL}/voice-references", timeout=10)
        new_count = response.json().get("total", 0)
        log_result("GET /voice-references (after create)", new_count > initial_count, f"Count: {new_count}")

        # 4. GET by ID
        response = requests.get(f"{BASE_URL}/voice-references/{voice_ref_id}", timeout=10)
        log_result(f"GET /voice-references/{voice_ref_id}", response.status_code == 200)

        # 5. GET audio file
        response = requests.get(f"{BASE_URL}/voice-references/{voice_ref_id}/audio", timeout=10)
        log_result(f"GET /voice-references/{voice_ref_id}/audio", response.status_code == 200 and len(response.content) > 0)

        return voice_ref_id

    except Exception as e:
        log_result("Voice Reference CRUD", False, str(e))
        return None


def test_voice_reference_delete(voice_ref_id: str):
    """Test DELETE voice reference."""
    if not voice_ref_id:
        log_result("DELETE /voice-references (skipped)", False, "No voice_ref_id")
        return False

    try:
        response = requests.delete(f"{BASE_URL}/voice-references/{voice_ref_id}", timeout=10)
        passed = response.status_code in [200, 204]
        log_result(f"DELETE /voice-references/{voice_ref_id}", passed, f"Status: {response.status_code}")

        # Verify deleted
        response = requests.get(f"{BASE_URL}/voice-references/{voice_ref_id}", timeout=10)
        log_result("GET deleted voice reference (should 404)", response.status_code == 404)

        return passed
    except Exception as e:
        log_result("DELETE /voice-references", False, str(e))
        return False


# =============================================================================
# Synthesis Endpoint Tests
# =============================================================================

def test_synthesis_crud(voice_ref_id: str = None):
    """Test full CRUD cycle for syntheses."""
    print("\n=== Testing Synthesis CRUD ===")
    synthesis_id = None

    try:
        # 1. LIST (initial)
        response = requests.get(f"{BASE_URL}/syntheses", timeout=10)
        initial_count = response.json().get("total", 0)
        log_result("GET /syntheses (initial)", response.status_code == 200, f"Count: {initial_count}")

        # 2. CREATE synthesis (without voice reference)
        print("\n--- Testing Synthesis Without Voice Reference ---")
        payload = {
            "text": "Hello, this is a test synthesis from the integration test suite.",
            "model": "turbo",
            "cfg_weight": 0.5,
            "exaggeration": 0.5
        }
        response = requests.post(f"{BASE_URL}/syntheses", json=payload, timeout=120)

        if response.status_code in [200, 201]:
            synthesis_id = response.json().get("id")
            status = response.json().get("status")
            duration = response.json().get("duration_seconds")
            log_result("POST /syntheses (turbo, no voice)", True, f"ID: {synthesis_id}, Status: {status}, Duration: {duration}s")
        else:
            log_result("POST /syntheses (turbo, no voice)", False, f"Status: {response.status_code}, Body: {response.text[:300]}")
            return None

        # 3. GET synthesis by ID
        response = requests.get(f"{BASE_URL}/syntheses/{synthesis_id}", timeout=10)
        log_result(f"GET /syntheses/{synthesis_id}", response.status_code == 200)

        # 4. GET audio file
        response = requests.get(f"{BASE_URL}/syntheses/{synthesis_id}/audio", timeout=10)
        audio_size = len(response.content) if response.status_code == 200 else 0
        log_result(f"GET /syntheses/{synthesis_id}/audio", response.status_code == 200 and audio_size > 0, f"Size: {audio_size} bytes")

        # 5. LIST (verify created)
        response = requests.get(f"{BASE_URL}/syntheses", timeout=10)
        new_count = response.json().get("total", 0)
        log_result("GET /syntheses (after create)", new_count > initial_count)

        # 6. Test with voice reference if available
        if voice_ref_id:
            print("\n--- Testing Synthesis With Voice Reference ---")
            payload = {
                "text": "This synthesis uses voice cloning from a reference audio.",
                "model": "turbo",
                "voice_reference_id": voice_ref_id,
                "cfg_weight": 0.5,
                "exaggeration": 0.5
            }
            response = requests.post(f"{BASE_URL}/syntheses", json=payload, timeout=120)
            if response.status_code in [200, 201]:
                voice_synthesis_id = response.json().get("id")
                log_result("POST /syntheses (with voice ref)", True, f"ID: {voice_synthesis_id}")

                # Delete this synthesis after test
                requests.delete(f"{BASE_URL}/syntheses/{voice_synthesis_id}", timeout=10)
            else:
                log_result("POST /syntheses (with voice ref)", False, f"Status: {response.status_code}")

        return synthesis_id

    except Exception as e:
        log_result("Synthesis CRUD", False, str(e))
        return None


def test_synthesis_with_different_models():
    """Test synthesis with different models."""
    print("\n=== Testing Different TTS Models ===")

    models_to_test = [
        ("turbo", None, None),
        ("standard", None, None),
        # ("multilingual", "en", None),  # Uncomment to test multilingual
    ]

    for model, language, _ in models_to_test:
        try:
            payload = {
                "text": f"Testing the {model} model.",
                "model": model,
                "cfg_weight": 0.5,
                "exaggeration": 0.5
            }
            if language:
                payload["language"] = language

            response = requests.post(f"{BASE_URL}/syntheses", json=payload, timeout=180)

            if response.status_code in [200, 201]:
                data = response.json()
                duration = data.get("duration_seconds", 0)
                processing_time = data.get("processing_time_seconds", 0)
                log_result(f"POST /syntheses (model={model})", True,
                          f"Duration: {duration:.2f}s, Processing: {processing_time:.2f}s")

                # Cleanup
                synthesis_id = data.get("id")
                if synthesis_id:
                    requests.delete(f"{BASE_URL}/syntheses/{synthesis_id}", timeout=10)
            else:
                log_result(f"POST /syntheses (model={model})", False,
                          f"Status: {response.status_code}, Body: {response.text[:200]}")
        except Exception as e:
            log_result(f"POST /syntheses (model={model})", False, str(e))


def test_synthesis_delete(synthesis_id: str):
    """Test DELETE synthesis."""
    if not synthesis_id:
        log_result("DELETE /syntheses (skipped)", False, "No synthesis_id")
        return False

    try:
        response = requests.delete(f"{BASE_URL}/syntheses/{synthesis_id}", timeout=10)
        passed = response.status_code in [200, 204]
        log_result(f"DELETE /syntheses/{synthesis_id}", passed, f"Status: {response.status_code}")

        # Verify deleted
        response = requests.get(f"{BASE_URL}/syntheses/{synthesis_id}", timeout=10)
        log_result("GET deleted synthesis (should 404)", response.status_code == 404)

        return passed
    except Exception as e:
        log_result("DELETE /syntheses", False, str(e))
        return False


def test_synthesis_validation():
    """Test synthesis input validation."""
    print("\n=== Testing Input Validation ===")

    # Test empty text
    response = requests.post(f"{BASE_URL}/syntheses", json={"text": "", "model": "turbo"}, timeout=10)
    log_result("POST /syntheses (empty text)", response.status_code in [400, 422])

    # Test invalid model
    response = requests.post(f"{BASE_URL}/syntheses", json={"text": "Hello", "model": "invalid_model"}, timeout=10)
    log_result("POST /syntheses (invalid model)", response.status_code in [400, 422])

    # Test invalid cfg_weight
    response = requests.post(f"{BASE_URL}/syntheses", json={"text": "Hello", "model": "turbo", "cfg_weight": 5.0}, timeout=10)
    log_result("POST /syntheses (invalid cfg_weight)", response.status_code in [400, 422])


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("CHATTERBOX TTS - FULL API INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Create test audio file
    create_test_audio()

    # Run tests
    test_health_endpoint()
    test_available_models()
    test_model_status()

    # Voice reference CRUD
    voice_ref_id = test_voice_reference_crud()

    # Synthesis CRUD (with voice reference for voice cloning test)
    synthesis_id = test_synthesis_crud(voice_ref_id)

    # Test different models
    test_synthesis_with_different_models()

    # Test validation
    test_synthesis_validation()

    # Cleanup
    print("\n=== Cleanup ===")
    test_synthesis_delete(synthesis_id)
    test_voice_reference_delete(voice_ref_id)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {results['passed'] + results['failed']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print("=" * 60)

    # Cleanup test file
    if TEST_AUDIO_FILE.exists():
        os.remove(TEST_AUDIO_FILE)
        print(f"Cleaned up: {TEST_AUDIO_FILE}")

    return results["failed"] == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
