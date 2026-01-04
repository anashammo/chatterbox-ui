#!/usr/bin/env python3
"""Run Docker Compose services"""
import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Docker Compose services")
    parser.add_argument("--build", action="store_true", help="Build images before starting")
    parser.add_argument("--detach", "-d", action="store_true", help="Run in detached mode")

    args = parser.parse_args()

    # Base command
    cmd = ["docker-compose", "up"]

    # Add options
    if args.build:
        cmd.append("--build")

    if args.detach:
        cmd.append("-d")

    # Run command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
