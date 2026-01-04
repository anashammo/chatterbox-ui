#!/usr/bin/env python3
"""Build Docker images for Chatterbox TTS system"""
import subprocess
import sys
import argparse

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        sys.exit(1)

    print(f"\n✅ Success: {description} completed")


def main():
    parser = argparse.ArgumentParser(description="Build Docker images")
    parser.add_argument("--backend", action="store_true", help="Build backend image only")
    parser.add_argument("--frontend", action="store_true", help="Build frontend image only")
    parser.add_argument("--no-cache", action="store_true", help="Build without cache")

    args = parser.parse_args()

    # If no specific service selected, build both
    build_all = not (args.backend or args.frontend)

    # Build backend
    if args.backend or build_all:
        cmd = ["docker", "build", "-t", "chatterbox-backend"]
        if args.no_cache:
            cmd.append("--no-cache")
        cmd.extend(["-f", "src/presentation/api/Dockerfile", "."])

        run_command(cmd, "Building backend image")

    # Build frontend
    if args.frontend or build_all:
        cmd = ["docker", "build", "-t", "chatterbox-frontend"]
        if args.no_cache:
            cmd.append("--no-cache")
        cmd.extend(["-f", "src/presentation/frontend/Dockerfile", "src/presentation/frontend"])

        run_command(cmd, "Building frontend image")

    print("\n" + "="*60)
    print("✅ All requested images built successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
