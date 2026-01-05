#!/usr/bin/env python3
"""Run Docker Compose services"""
import subprocess
import sys
import argparse
import os
from pathlib import Path


def get_ngrok_token() -> str:
    """Get NGROK_AUTHTOKEN from environment or .env file"""
    # First check environment
    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        return token

    # Then check .env file
    env_file = Path(__file__).parent.parent.parent / "src" / "presentation" / "api" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("NGROK_AUTHTOKEN="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token
    return ""


def main():
    parser = argparse.ArgumentParser(description="Run Docker Compose services")
    parser.add_argument("--build", action="store_true", help="Build images before starting")
    parser.add_argument("--detach", "-d", action="store_true", default=True, help="Run in detached mode (default)")
    parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground (override detach)")
    parser.add_argument("--no-ngrok", action="store_true", help="Exclude ngrok tunnels (ngrok is included by default)")

    args = parser.parse_args()

    # Base command
    cmd = ["docker", "compose"]

    # Add ngrok profile by default (unless --no-ngrok is specified)
    if not args.no_ngrok:
        # Check if NGROK_AUTHTOKEN is available (env or .env file)
        if not get_ngrok_token():
            print("Error: NGROK_AUTHTOKEN not found in environment or src/presentation/api/.env")
            print("\nTo set it, add to src/presentation/api/.env:")
            print("  NGROK_AUTHTOKEN=your_token_here")
            print("\nOr use --no-ngrok to start without ngrok tunnels.")
            sys.exit(1)
        cmd.extend(["--profile", "ngrok"])
        print("Including ngrok tunnels:")
        print("  Backend:  https://anas-hammo-chatterbox-backend.ngrok.dev")
        print("  Frontend: https://anas-hammo-chatterbox-frontend.ngrok.dev")
        print()
    else:
        print("Starting without ngrok tunnels (--no-ngrok specified)")

    cmd.append("up")

    # Add options
    if args.build:
        cmd.append("--build")

    if args.detach and not args.foreground:
        cmd.append("-d")

    # Run command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0 and not args.no_ngrok:
        print("\nNgrok web interfaces:")
        print("  Backend:  http://localhost:4040")
        print("  Frontend: http://localhost:4041")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
