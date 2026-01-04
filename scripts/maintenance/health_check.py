"""
System Health Check - Comprehensive system verification for Chatterbox TTS

Verifies the entire Chatterbox TTS system is configured correctly
and ready for development or production use.

Usage:
    python scripts/maintenance/health_check.py
    python scripts/maintenance/health_check.py --verbose
"""

import sys
import os
import subprocess
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.terminal import Colors


class HealthChecker:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks_passed = 0
        self.checks_total = 0
        self.warnings = []
        self.errors = []

    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

    def check(self, name: str, passed: bool, message: str = "", is_warning: bool = False):
        """Record and print check result"""
        self.checks_total += 1

        if passed:
            self.checks_passed += 1
            status = f"{Colors.GREEN}[OK]{Colors.RESET}"
        elif is_warning:
            status = f"{Colors.YELLOW}[WARN]{Colors.RESET}"
            self.warnings.append(f"{name}: {message}")
        else:
            status = f"{Colors.RED}[FAIL]{Colors.RESET}"
            self.errors.append(f"{name}: {message}")

        print(f"{status} {name}")
        if message and (self.verbose or not passed):
            indent = "  "
            color = Colors.YELLOW if is_warning else (Colors.GREEN if passed else Colors.RED)
            print(f"{indent}{color}{message}{Colors.RESET}")

    def run_command(self, cmd: List[str]) -> Tuple[int, str, str]:
        """Run shell command"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=10
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def check_python_version(self):
        """Check Python version"""
        version = sys.version_info
        passed = version.major == 3 and version.minor >= 9
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        self.check(
            "Python version",
            passed,
            f"Found Python {version_str} (requires 3.9+)",
            is_warning=not passed
        )

    def check_git_repository(self):
        """Check if in git repository"""
        returncode, _, _ = self.run_command(['git', 'rev-parse', '--git-dir'])
        passed = returncode == 0

        if passed:
            returncode, branch, _ = self.run_command(['git', 'branch', '--show-current'])
            branch_name = branch.strip() if returncode == 0 else "unknown"
            self.check("Git repository", passed, f"Current branch: {branch_name}")
        else:
            self.check("Git repository", passed, "Not a git repository")

    def check_database(self):
        """Check database exists and is accessible"""
        db_path = project_root / "chatterbox_tts.db"

        if not db_path.exists():
            self.check("Database file", False, "Database not found (run: python scripts/setup/init_db.py)")
            return

        # Check if we can connect
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Check tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                required_tables = ['syntheses', 'voice_references']
                missing_tables = [t for t in required_tables if t not in tables]

                if missing_tables:
                    self.check("Database schema", False, f"Missing tables: {', '.join(missing_tables)}")
                else:
                    # Count records
                    cursor.execute("SELECT COUNT(*) FROM syntheses")
                    synth_count = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM voice_references")
                    voice_count = cursor.fetchone()[0]

                    size_mb = db_path.stat().st_size / (1024 * 1024)

                    self.check(
                        "Database",
                        True,
                        f"{synth_count} syntheses, {voice_count} voice references ({size_mb:.2f} MB)"
                    )
        except Exception as e:
            self.check("Database access", False, str(e))

    def check_audio_directories(self):
        """Check audio output and voice reference directories"""
        audio_dir = project_root / "audio_outputs"
        voice_dir = project_root / "voice_references"

        # Check audio outputs directory
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.*"))
            total_size_mb = sum(f.stat().st_size for f in audio_files if f.is_file()) / (1024 * 1024)
            self.check(
                "Audio outputs directory",
                True,
                f"{len(audio_files)} files ({total_size_mb:.1f} MB)"
            )
        else:
            self.check(
                "Audio outputs directory",
                False,
                "Directory does not exist",
                is_warning=True
            )

        # Check voice references directory
        if voice_dir.exists():
            voice_files = list(voice_dir.glob("*.*"))
            total_size_mb = sum(f.stat().st_size for f in voice_files if f.is_file()) / (1024 * 1024)
            self.check(
                "Voice references directory",
                True,
                f"{len(voice_files)} files ({total_size_mb:.1f} MB)"
            )
        else:
            self.check(
                "Voice references directory",
                False,
                "Directory does not exist",
                is_warning=True
            )

    def check_python_dependencies(self):
        """Check Python dependencies"""
        requirements_file = project_root / "src" / "presentation" / "api" / "requirements.txt"

        if not requirements_file.exists():
            self.check("Requirements file", False, "requirements.txt not found")
            return

        # Check if virtual environment is active
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

        self.check(
            "Virtual environment",
            in_venv,
            "Active" if in_venv else "Not activated (recommended to use venv)",
            is_warning=not in_venv
        )

        # Check key packages
        key_packages = ['fastapi', 'uvicorn', 'sqlalchemy', 'torch', 'torchaudio']

        for package in key_packages:
            try:
                __import__(package.replace('-', '_'))
                self.check(f"Package: {package}", True, "Installed")
            except ImportError:
                self.check(f"Package: {package}", False, "Not installed")

    def check_node_dependencies(self):
        """Check Node.js and frontend dependencies"""
        frontend_dir = project_root / "src" / "presentation" / "frontend"
        node_modules = frontend_dir / "node_modules"

        # Check Node.js version
        returncode, stdout, _ = self.run_command(['node', '--version'])
        if returncode == 0:
            node_version = stdout.strip()
            self.check("Node.js", True, f"Version {node_version}")
        else:
            self.check("Node.js", False, "Not installed or not in PATH")
            return

        # Check npm
        returncode, stdout, _ = self.run_command(['npm', '--version'])
        if returncode == 0:
            npm_version = stdout.strip()
            self.check("npm", True, f"Version {npm_version}")
        else:
            self.check("npm", False, "Not installed")

        # Check frontend dependencies
        if node_modules.exists():
            # Count installed packages
            package_count = len(list(node_modules.iterdir()))
            self.check("Frontend dependencies", True, f"{package_count} packages installed")
        else:
            self.check(
                "Frontend dependencies",
                False,
                "Not installed (run: cd src/presentation/frontend && npm install)"
            )

    def check_chatterbox_models(self):
        """Check Chatterbox TTS models in cache"""
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        if not cache_dir.exists():
            self.check(
                "Model cache",
                False,
                "No models downloaded (models are downloaded on first use)",
                is_warning=True
            )
            return

        # Check for Chatterbox models
        chatterbox_models = list(cache_dir.glob("models--ResembleAI--chatterbox*"))

        if chatterbox_models:
            total_size_mb = 0
            for model_dir in chatterbox_models:
                for f in model_dir.rglob("*"):
                    if f.is_file():
                        total_size_mb += f.stat().st_size / (1024 * 1024)

            self.check(
                "Chatterbox models",
                True,
                f"{len(chatterbox_models)} model(s) cached ({total_size_mb:.0f} MB)"
            )
        else:
            self.check(
                "Chatterbox models",
                False,
                "No models found (will download on first synthesis)",
                is_warning=True
            )

    def check_gpu(self):
        """Check GPU availability"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()

            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                self.check(
                    "GPU",
                    True,
                    f"{gpu_name} ({gpu_memory:.1f} GB VRAM)"
                )
            else:
                self.check(
                    "GPU",
                    False,
                    "CUDA not available (will use CPU, synthesis will be slower)",
                    is_warning=True
                )
        except ImportError:
            self.check("GPU", False, "PyTorch not installed")

    def check_backend_server(self):
        """Check if backend server is running"""
        try:
            response = requests.get("http://localhost:8002/docs", timeout=2)
            if response.status_code == 200:
                self.check("Backend server", True, "Running on http://localhost:8002")
            else:
                self.check("Backend server", False, f"Unexpected status: {response.status_code}")
        except requests.exceptions.RequestException:
            self.check(
                "Backend server",
                False,
                "Not running (start with: python scripts/server/run_backend.py)",
                is_warning=True
            )

    def check_frontend_server(self):
        """Check if frontend server is running"""
        try:
            response = requests.get("http://localhost:4201", timeout=2)
            if response.status_code == 200:
                self.check("Frontend server", True, "Running on http://localhost:4201")
            else:
                self.check("Frontend server", False, f"Unexpected status: {response.status_code}")
        except requests.exceptions.RequestException:
            self.check(
                "Frontend server",
                False,
                "Not running (start with: python scripts/server/run_frontend.py)",
                is_warning=True
            )

    def check_disk_space(self):
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(project_root)

            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            used_percent = (used / total) * 100

            passed = free_gb > 5.0  # At least 5GB free
            self.check(
                "Disk space",
                passed,
                f"{free_gb:.1f} GB free of {total_gb:.1f} GB ({used_percent:.1f}% used)",
                is_warning=not passed
            )
        except Exception as e:
            self.check("Disk space", False, str(e), is_warning=True)

    def run_all_checks(self):
        """Run all health checks"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Chatterbox TTS System Health Check{Colors.RESET}")
        print(f"{Colors.BLUE}Project: {project_root}{Colors.RESET}\n")

        self.print_header("1. Environment")
        self.check_python_version()
        self.check_git_repository()
        self.check_disk_space()

        self.print_header("2. Dependencies")
        self.check_python_dependencies()
        self.check_node_dependencies()

        self.print_header("3. GPU & Models")
        self.check_gpu()
        self.check_chatterbox_models()

        self.print_header("4. Data & Storage")
        self.check_database()
        self.check_audio_directories()

        self.print_header("5. Servers (Optional)")
        self.check_backend_server()
        self.check_frontend_server()

        # Summary
        return self.print_summary()

    def print_summary(self):
        """Print health check summary"""
        self.print_header("Health Check Summary")

        pass_rate = (self.checks_passed / self.checks_total * 100) if self.checks_total > 0 else 0

        # Determine status
        if pass_rate == 100 and not self.warnings:
            status_color = Colors.GREEN
            status_text = "EXCELLENT"
            status_icon = "[OK]"
        elif pass_rate >= 90:
            status_color = Colors.GREEN
            status_text = "GOOD"
            status_icon = "[OK]"
        elif pass_rate >= 70:
            status_color = Colors.YELLOW
            status_text = "FAIR"
            status_icon = "[WARN]"
        else:
            status_color = Colors.RED
            status_text = "POOR"
            status_icon = "[FAIL]"

        print(f"{status_color}{Colors.BOLD}{status_icon} System Health: {status_text}{Colors.RESET}")
        print(f"Checks Passed: {Colors.GREEN}{self.checks_passed}/{self.checks_total}{Colors.RESET} ({pass_rate:.1f}%)")

        if self.warnings:
            print(f"\n{Colors.YELLOW}Warnings ({len(self.warnings)}):{Colors.RESET}")
            for warning in self.warnings:
                print(f"  ! {warning}")

        if self.errors:
            print(f"\n{Colors.RED}Errors ({len(self.errors)}):{Colors.RESET}")
            for error in self.errors:
                print(f"  X {error}")

        print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

        return pass_rate >= 70  # Success if 70% or more checks pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Chatterbox TTS system health check')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    checker = HealthChecker(verbose=args.verbose)
    success = checker.run_all_checks()

    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}ERROR: {str(e)}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
