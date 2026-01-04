"""
Automated testing script for UI enhancement features

Tests backend endpoints and validates responses without browser interaction.
Verifies the new features implemented in the UI enhancements branch.

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
API_BASE_URL = "http://localhost:8001/api/v1"
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


def test_backend_health() -> bool:
    """Test if backend server is running"""
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/docs", timeout=TIMEOUT)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def test_api_endpoints() -> List[Tuple[str, bool, str]]:
    """Test core API endpoints are accessible"""
    results = []

    endpoints = [
        ("GET /docs", "/docs", 200),
        ("GET /api/v1/transcriptions", f"{API_BASE_URL}/transcriptions", 200),
        ("GET /api/v1/models/available", f"{API_BASE_URL}/models/available", 200),
    ]

    for name, url, expected_status in endpoints:
        try:
            full_url = url if url.startswith('http') else f"{API_BASE_URL.replace('/api/v1', '')}{url}"
            response = requests.get(full_url, timeout=TIMEOUT)
            passed = response.status_code == expected_status
            message = f"Status: {response.status_code}" if not passed else ""
            results.append((name, passed, message))
        except requests.exceptions.RequestException as e:
            results.append((name, False, f"Error: {str(e)}"))

    return results


def test_audio_download_endpoint() -> Tuple[bool, str]:
    """
    Test audio download endpoint structure (Feature #1)
    Cannot test actual .webm -> .wav conversion without a real transcription
    """
    try:
        # Just verify the endpoint exists by checking API docs
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/docs", timeout=TIMEOUT)
        if response.status_code == 200:
            # Check if the endpoint is documented
            if "transcriptions/{transcription_id}/audio" in response.text:
                return True, "Audio download endpoint documented in API"
            else:
                return False, "Audio download endpoint not found in API docs"
        return False, f"Could not access API docs (status: {response.status_code})"
    except requests.exceptions.RequestException as e:
        return False, f"Error: {str(e)}"


def check_database_exists() -> Tuple[bool, str]:
    """Check if database file exists"""
    db_path = project_root / "whisper_transcriptions.db"
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        return True, f"Database exists ({size_mb:.2f} MB)"
    return False, "Database file not found"


def check_uploads_directory() -> Tuple[bool, str]:
    """Check if uploads directory exists"""
    uploads_path = project_root / "uploads"
    if uploads_path.exists():
        # Count subdirectories (audio file groups)
        subdirs = [d for d in uploads_path.iterdir() if d.is_dir()]
        return True, f"Uploads directory exists ({len(subdirs)} audio file groups)"
    return False, "Uploads directory not found"


def check_frontend_build() -> Tuple[bool, str]:
    """Check if frontend is built"""
    frontend_path = project_root / "src" / "presentation" / "frontend"
    node_modules = frontend_path / "node_modules"

    if not node_modules.exists():
        return False, "node_modules not found (run: cd src/presentation/frontend && npm install)"

    return True, "Frontend dependencies installed"


def check_whisper_models() -> Tuple[bool, str]:
    """Check if Whisper models are downloaded"""
    cache_dir = Path.home() / ".cache" / "whisper"

    if not cache_dir.exists():
        return False, "Whisper cache directory not found (no models downloaded)"

    # Check for common model files
    models = ["tiny.pt", "base.pt", "small.pt", "medium.pt", "large-v3.pt", "large-v3-turbo.pt"]
    found_models = [m for m in models if (cache_dir / m).exists()]

    if found_models:
        return True, f"Found {len(found_models)} model(s): {', '.join([m.replace('.pt', '') for m in found_models])}"

    return False, "No Whisper models found in cache"


def run_all_tests():
    """Run all automated tests"""
    print_header("Automated Feature Testing")

    total_tests = 0
    passed_tests = 0

    # Test 1: Backend Health
    print_header("1. Backend Server Health")
    is_healthy = test_backend_health()
    print_test("Backend server is running", is_healthy,
               "Server accessible at http://localhost:8001" if is_healthy else "Cannot reach backend server")
    total_tests += 1
    if is_healthy:
        passed_tests += 1

    # Test 2: API Endpoints
    print_header("2. Core API Endpoints")
    endpoint_results = test_api_endpoints()
    for name, passed, message in endpoint_results:
        print_test(name, passed, message)
        total_tests += 1
        if passed:
            passed_tests += 1

    # Test 3: Audio Download Endpoint (Feature #1)
    print_header("3. Audio Download Endpoint (Feature #1)")
    passed, message = test_audio_download_endpoint()
    print_test("Audio download endpoint exists", passed, message)
    print(f"       {Colors.YELLOW}Note: .webm -> .wav conversion requires manual testing with actual file{Colors.RESET}")
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 4: Database
    print_header("4. Database Status")
    passed, message = check_database_exists()
    print_test("Database file exists", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 5: Uploads Directory
    print_header("5. File Storage")
    passed, message = check_uploads_directory()
    print_test("Uploads directory exists", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 6: Frontend Build
    print_header("6. Frontend Status")
    passed, message = check_frontend_build()
    print_test("Frontend dependencies installed", passed, message)
    total_tests += 1
    if passed:
        passed_tests += 1

    # Test 7: Whisper Models
    print_header("7. Whisper Models")
    passed, message = check_whisper_models()
    print_test("Whisper models downloaded", passed, message)
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
    print(f"{Colors.CYAN}Manual Testing Required:{Colors.RESET}")
    print("  - Feature #2: Audio controls in header (History view)")
    print("  - Feature #3: LLM enhancement badges (History view)")
    print("  - Feature #4: Enhanced text preview (History view)")
    print("  - Feature #5-8: Details view changes (readonly, badges, styles)")
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
