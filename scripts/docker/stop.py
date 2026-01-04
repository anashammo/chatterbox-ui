#!/usr/bin/env python3
"""Stop Docker Compose services"""
import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Stop Docker Compose services")
    parser.add_argument("--remove-volumes", "-v", action="store_true",
                       help="Remove volumes as well (WARNING: deletes data)")

    args = parser.parse_args()

    cmd = ["docker-compose", "down"]

    if args.remove_volumes:
        print("⚠️  WARNING: This will remove all volumes and delete data!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            sys.exit(0)
        cmd.append("-v")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
