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
    parser.add_argument("--ngrok", action="store_true", help="Also start ngrok tunnels after rebuild")

    args = parser.parse_args()

    # Check ngrok auth token if ngrok requested
    if args.ngrok and not os.environ.get("NGROK_AUTHTOKEN"):
        print("Error: NGROK_AUTHTOKEN environment variable is not set.")
        print("\nTo set it:")
        print("  Windows:  set NGROK_AUTHTOKEN=your_token_here")
        print("  Linux:    export NGROK_AUTHTOKEN=your_token_here")
        sys.exit(1)

    # Build base commands
    base_cmd = ["docker", "compose"]
    if args.ngrok:
        base_cmd.extend(["--profile", "ngrok"])

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
        if args.ngrok:
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
