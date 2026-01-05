"""
Automated testing script for Chatterbox TTS features

Tests backend endpoints and validates responses without browser interaction.
Verifies core TTS functionality and system setup.

Usage:
    python scripts/testing/test_features.py
"""

import sys
import os
import requests
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.terminal import Colors

# Test configuration
API_BASE_URL = "http://localhost:8002/api/v1"
TIMEOUT = 10  # seconds


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def print_test(name: str, passed: bool, message: str = ""):
    """Print test result"""
    status = f"{Colors.GREEN}[PASS]{Colors.RESET}" if passed else f"{Colors.RED}[FAIL]{Colors.RESET}"
    print(f"{status} | {name}")
    if message:
        print(f"       {Colors.YELLOW}{message}{Colors.RESET}")


def test_backend_health() -> Tuple[bool, str]:
    """Test if backend server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            return True, "Backend healthy at http://localhost:8002"
        return False, f"Health check returned status {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Cannot reach backend server: {str(e)}"


def test_api_endpoints() -> List[Tuple[str, bool, str]]:
    """Test core API endpoints are accessible"""
    results = []

    endpoints = [
        ("GET /docs", "http://localhost:8002/docs", 200),
        ("GET /api/v1/health", f"{API_BASE_URL}/health", 200),
        ("GET /api/v1/syntheses", f"{API_BASE_URL}/syntheses", 200),
        ("GET /api/v1/voice-references", f"{API_BASE_URL}/voice-references", 200),
        ("GET /api/v1/models/available", f"{API_BASE_URL}/models/available", 200),
    ]

    for name, url, expected_status in endpoints:
        try:
            response = requests.get(url, timeout=TIMEOUT)
            passed = response.status_code == expected_status
            message = f"Status: {response.status_code}" if not passed else ""
            results.append((name, passed, message))
        except requests.exceptions.RequestException as e:
            results.append((name, False, f"Error: {str(e)}"))

    return results


def test_models_endpoint() -> Tuple[bool, str]:
    """Test models endpoint returns valid model list"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/available", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            # Handle both {"models": [...]} and [...] formats
            models = data.get('models', data) if isinstance(data, dict) else data
            if isinstance(models, list) and len(models) > 0:
                model_names = [m.get('name', m) if isinstance(m, dict) else m for m in models]
                return True, f"Available models: {', '.join(model_names)}"
            return False, "No models returned"
        return False, f"Status: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Error: {str(e)}"


def check_database_exists() -> Tuple[bool, str]:
    """Check if database is accessible (via API or local file)"""
    # First try API - works for both Docker (PostgreSQL) and local (SQLite)
    try:
        response = requests.get(f"{API_BASE_URL}/syntheses?limit=1", timeout=TIMEOUT)
        if response.status_code == 200:
            return True, "Database accessible via API (Docker PostgreSQL or local SQLite)"
    except requests.exceptions.RequestException:
        pass

    # Fallback: check local SQLite file (local development only)
    db_path = project_root / "chatterbox_tts.db"
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        return True, f"SQLite database exists ({size_kb:.2f} KB)"
    return False, "Database not accessible (API check failed, no local SQLite)"


def check_audio_outputs_directory() -> Tuple[bool, str]:
    """Check if audio outputs directory exists"""
    audio_path = project_root / "audio_outputs"
    if audio_path.exists():
        files = list(audio_path.glob("*.wav"))
        return True, f"Audio outputs directory exists ({len(files)} .wav files)"
    return False, "Audio outputs directory not found (will be created on first synthesis)"


def check_voice_references_directory() -> Tuple[bool, str]:
    """Check if voice references directory exists"""
    voice_ref_path = project_root / "voice_references"
    if voice_ref_path.exists():
        files = list(voice_ref_path.iterdir())
        return True, f"Voice references directory exists ({len(files)} files)"
    return False, "Voice references directory not found (will be created on first upload)"


def check_frontend_build() -> Tuple[bool, str]:
    """Check if frontend dependencies are installed"""
    frontend_path = project_root / "src" / "presentation" / "frontend"
    node_modules = frontend_path / "node_modules"

    if not node_modules.exists():
        return False, "node_modules not found (run: cd src/presentation/frontend && npm install)"

    return True, "Frontend dependencies installed"


def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists with required settings"""
    env_path = project_root / "src" / "presentation" / "api" / ".env"

    if not env_path.exists():
        return False, ".env not found (copy from .env.example)"

    # Check for HF_TOKEN
    content = env_path.read_text()
    if "HF_TOKEN=" in content:
        # Check if it has a value
        for line in content.split('\n'):
            if line.startswith('HF_TOKEN=') and len(line.split('=', 1)[1].strip()) > 0:
                return True, ".env exists with HF_TOKEN configured"
        return False, ".env exists but HF_TOKEN is empty"

    return False, ".env exists but missing HF_TOKEN"


def check_chatterbox_models() -> Tuple[bool, str]:
    """Check if Chatterbox models are available (via API or local cache)"""
    # First check via API - works for Docker where models are in container
    try:
        response = requests.get(f"{API_BASE_URL}/models/available", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', data) if isinstance(data, dict) else data
            if isinstance(models, list) and len(models) > 0:
                model_names = [m.get('name', 'unknown') for m in models]
                return True, f"Models available via API: {', '.join(model_names)}"
    except requests.exceptions.RequestException:
        pass

    # Fallback: check local HuggingFace cache (local development only)
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        chatterbox_dirs = list(cache_dir.glob("models--ResembleAI--chatterbox*"))
        if chatterbox_dirs:
            model_names = [d.name.replace("models--ResembleAI--", "") for d in chatterbox_dirs]
            return True, f"Found {len(chatterbox_dirs)} local model(s): {', '.join(model_names)}"

    return False, "Models not accessible (API check failed, no local cache)"


def run_all_tests():
    """Run all automated tests"""
    print_header("Chatterbox TTS - Automated Testing")

    total_tests = 0
    passed_tests = 0

    # Test 1: Environment Setup
    print_header("1. Environment Setup")
    passed, message = check_env_file()
    print_test(".env file configured", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 2: Backend Health
    print_header("2. Backend Server Health")
    passed, message = test_backend_health()
    print_test("Backend server is running", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 3: API Endpoints
    print_header("3. Core API Endpoints")
    endpoint_results = test_api_endpoints()
    for name, passed, message in endpoint_results:
        print_test(name, passed, message)
        total_tests += 1
        if passed:
            passed_tests += 1

    # Test 4: Models Endpoint
    print_header("4. TTS Models")
    passed, message = test_models_endpoint()
    print_test("Models endpoint returns data", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 5: Database
    print_header("5. Database Status")
    passed, message = check_database_exists()
    print_test("Database file exists", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 6: File Storage
    print_header("6. File Storage")
    passed, message = check_audio_outputs_directory()
    print_test("Audio outputs directory", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    passed, message = check_voice_references_directory()
    print_test("Voice references directory", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 7: Frontend
    print_header("7. Frontend Status")
    passed, message = check_frontend_build()
    print_test("Frontend dependencies installed", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 8: Chatterbox Models
    print_header("8. Chatterbox Models Cache")
    passed, message = check_chatterbox_models()
    print_test("Chatterbox models downloaded", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Summary
    print_header("Test Summary")
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    if pass_rate == 100:
        color = Colors.GREEN
        status = "ALL TESTS PASSED"
    elif pass_rate >= 70:
        color = Colors.YELLOW
        status = "MOST TESTS PASSED"
    else:
        color = Colors.RED
        status = "MANY TESTS FAILED"

    print(f"{color}{Colors.BOLD}{status}{Colors.RESET}")
    print(f"Passed: {Colors.GREEN}{passed_tests}/{total_tests}{Colors.RESET} ({pass_rate:.1f}%)")

    if passed_tests < total_tests:
        print(f"\n{Colors.YELLOW}[!] Some tests failed. Please review the output above.{Colors.RESET}")
        print(f"{Colors.YELLOW}Note: Backend must be running for API tests to pass.{Colors.RESET}")

    print("\n" + "=" * 60)
    print(f"{Colors.CYAN}Manual Testing Recommended:{Colors.RESET}")
    print("  - Text-to-speech synthesis with different models")
    print("  - Voice reference upload and voice cloning")
    print("  - Audio playback and download")
    print("  - Synthesis history and management")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

    return passed_tests == total_tests


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Testing interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}ERROR: {str(e)}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
