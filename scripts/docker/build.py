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
        print(f"\n[X] Error: {description} failed")
        sys.exit(1)

    print(f"\n[OK] Success: {description} completed")


def main():
    parser = argparse.ArgumentParser(description="Build Docker images")
    parser.add_argument("--backend", action="store_true", help="Build backend image only")
    parser.add_argument("--frontend", action="store_true", help="Build frontend image only")
    parser.add_argument("--no-cache", action="store_true", help="Build without cache")

    args = parser.parse_args()

    # If no specific service selected, build all
    build_all = not (args.backend or args.frontend)

    # Base command using docker compose
    base_cmd = ["docker", "compose", "build"]

    if args.no_cache:
        base_cmd.append("--no-cache")

    # Build specific services or all
    if build_all:
        run_command(base_cmd, "Building all images")
    else:
        services = []
        if args.backend:
            services.append("backend")
        if args.frontend:
            services.append("frontend")

        cmd = base_cmd + services
        run_command(cmd, f"Building {', '.join(services)} image(s)")

    print("\n" + "="*60)
    print("[OK] All requested images built successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
