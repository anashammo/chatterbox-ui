#!/usr/bin/env python3
"""Install UV package manager

This script installs the UV package manager, a fast Python package installer
written in Rust that serves as a drop-in replacement for pip.

UV provides 10-100x faster package installation compared to pip through:
- Parallel downloads
- Global package cache
- Zero-copy installations (when possible)
"""
import sys
import subprocess
import shutil
import platform

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    """Install UV package manager"""
    print("=" * 60)
    print("UV Package Manager Installation")
    print("=" * 60)
    print()

    # Check if UV is already installed
    if shutil.which('uv'):
        print("✓ UV is already installed")
        print()
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        print(result.stdout.strip())
        print()
        print("To upgrade UV, run: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return

    print("Installing UV package manager...")
    print()

    try:
        system = platform.system()

        if system == 'Windows':
            # Windows installation via PowerShell
            print("Platform: Windows")
            print("Running: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
            print()

            result = subprocess.run([
                'powershell', '-Command',
                'irm https://astral.sh/uv/install.ps1 | iex'
            ], check=True, capture_output=True, text=True)

            print(result.stdout)

        elif system in ['Linux', 'Darwin']:
            # Linux/Mac installation via shell
            print(f"Platform: {system}")
            print("Running: curl -LsSf https://astral.sh/uv/install.sh | sh")
            print()

            result = subprocess.run([
                'sh', '-c',
                'curl -LsSf https://astral.sh/uv/install.sh | sh'
            ], check=True)

        else:
            print(f"❌ Unsupported platform: {system}")
            print("Please install UV manually: https://docs.astral.sh/uv/getting-started/installation/")
            sys.exit(1)

        print()
        print("=" * 60)
        print("✅ UV installed successfully!")
        print("=" * 60)
        print()

        # Try to verify installation
        if shutil.which('uv'):
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
            print(result.stdout.strip())
        else:
            print("⚠️  UV installed but not in PATH")
            print("You may need to restart your terminal or add UV to your PATH")
            print()
            if system == 'Windows':
                print("UV is installed at: C:\\Users\\<username>\\.local\\bin")
            else:
                print("UV is installed at: ~/.local/bin")

        print()
        print("Usage:")
        print("  uv pip install <package>    # Install package")
        print("  uv pip install -r requirements.txt  # Install from requirements")
        print("  uv --help                   # Show help")
        print()

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("❌ Installation failed")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Please try manual installation:")
        print("  Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        print("  Linux/Mac: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print()
        print("Documentation: https://docs.astral.sh/uv/")
        sys.exit(1)


if __name__ == "__main__":
    main()
