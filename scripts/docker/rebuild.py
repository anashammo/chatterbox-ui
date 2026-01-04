#!/usr/bin/env python3
"""Rebuild and restart Docker containers"""
import subprocess
import sys

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    print("Stopping containers...")
    subprocess.run(["docker-compose", "down"])

    print("\nRebuilding images...")
    result = subprocess.run(["docker-compose", "build"])

    if result.returncode != 0:
        print("\n[X] Build failed!")
        sys.exit(1)

    print("\nStarting containers...")
    result = subprocess.run(["docker-compose", "up", "-d"])

    if result.returncode == 0:
        print("\n[✓] Rebuild and restart complete!")
        print("\nView logs with: python scripts/docker/logs.py -f")
    else:
        print("\n[X] Failed to start containers!")
        sys.exit(1)


if __name__ == "__main__":
    main()
