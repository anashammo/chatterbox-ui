#!/usr/bin/env python3
"""Stop Docker Compose services"""
import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Stop Docker Compose services")
    parser.add_argument("--remove-volumes", "-v", action="store_true",
                       help="Remove volumes as well (WARNING: deletes data)")
    parser.add_argument("--no-ngrok", action="store_true",
                       help="Exclude ngrok tunnels from stop (ngrok is included by default)")

    args = parser.parse_args()

    # Base command
    cmd = ["docker", "compose"]

    # Add ngrok profile by default (unless --no-ngrok is specified)
    if not args.no_ngrok:
        cmd.extend(["--profile", "ngrok"])
        print("Including ngrok tunnels in stop command...")
    else:
        print("Stopping without ngrok tunnels (--no-ngrok specified)")

    cmd.append("down")

    if args.remove_volumes:
        print("WARNING: This will remove all volumes and delete data!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            sys.exit(0)
        cmd.append("-v")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
