#!/usr/bin/env python3
"""Rebuild and restart Docker containers"""
import subprocess
import sys
import argparse
import os

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Rebuild and restart Docker containers")
    parser.add_argument("--no-ngrok", action="store_true", help="Exclude ngrok tunnels (ngrok is included by default)")

    args = parser.parse_args()

    # Check ngrok auth token (required by default unless --no-ngrok)
    if not args.no_ngrok and not os.environ.get("NGROK_AUTHTOKEN"):
        print("Error: NGROK_AUTHTOKEN environment variable is not set.")
        print("\nTo set it:")
        print("  Windows:  set NGROK_AUTHTOKEN=your_token_here")
        print("  Linux:    export NGROK_AUTHTOKEN=your_token_here")
        print("\nOr use --no-ngrok to rebuild without ngrok tunnels.")
        sys.exit(1)

    # Build base commands
    base_cmd = ["docker", "compose"]
    if not args.no_ngrok:
        base_cmd.extend(["--profile", "ngrok"])
    else:
        print("Rebuilding without ngrok tunnels (--no-ngrok specified)")

    print("Stopping containers...")
    subprocess.run(base_cmd + ["down"])

    print("\nRebuilding images...")
    result = subprocess.run(["docker", "compose", "build"])

    if result.returncode != 0:
        print("\n[X] Build failed!")
        sys.exit(1)

    print("\nStarting containers...")
    result = subprocess.run(base_cmd + ["up", "-d"])

    if result.returncode == 0:
        print("\n[OK] Rebuild and restart complete!")
        print("\nView logs with: python scripts/docker/logs.py -f")
        if not args.no_ngrok:
            print("\nNgrok tunnels:")
            print("  Backend:  https://anas-hammo-chatterbox-backend.ngrok.dev")
            print("  Frontend: https://anas-hammo-chatterbox-frontend.ngrok.dev")
            print("\nNgrok web interfaces:")
            print("  Backend:  http://localhost:4040")
            print("  Frontend: http://localhost:4041")
    else:
        print("\n[X] Failed to start containers!")
        sys.exit(1)


if __name__ == "__main__":
    main()
