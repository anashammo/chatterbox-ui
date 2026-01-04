#!/usr/bin/env python3
"""Run Docker Compose services"""
import subprocess
import sys
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Run Docker Compose services")
    parser.add_argument("--build", action="store_true", help="Build images before starting")
    parser.add_argument("--detach", "-d", action="store_true", default=True, help="Run in detached mode (default)")
    parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground (override detach)")
    parser.add_argument("--ngrok", action="store_true", help="Also start ngrok tunnels")

    args = parser.parse_args()

    # Base command
    cmd = ["docker", "compose"]

    # Add ngrok profile if requested
    if args.ngrok:
        # Check if NGROK_AUTHTOKEN is set
        if not os.environ.get("NGROK_AUTHTOKEN"):
            print("Error: NGROK_AUTHTOKEN environment variable is not set.")
            print("\nTo set it:")
            print("  Windows:  set NGROK_AUTHTOKEN=your_token_here")
            print("  Linux:    export NGROK_AUTHTOKEN=your_token_here")
            sys.exit(1)
        cmd.extend(["--profile", "ngrok"])
        print("Including ngrok tunnels:")
        print("  Backend:  https://anas-hammo-chatterbox-backend.ngrok.dev")
        print("  Frontend: https://anas-hammo-chatterbox-frontend.ngrok.dev")
        print()

    cmd.append("up")

    # Add options
    if args.build:
        cmd.append("--build")

    if args.detach and not args.foreground:
        cmd.append("-d")

    # Run command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0 and args.ngrok:
        print("\nNgrok web interfaces:")
        print("  Backend:  http://localhost:4040")
        print("  Frontend: http://localhost:4041")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
