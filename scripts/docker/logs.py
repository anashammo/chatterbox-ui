#!/usr/bin/env python3
"""View Docker container logs"""
import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="View Docker container logs")
    parser.add_argument("service", nargs="?",
                       help="Service name (backend, frontend, postgres, ngrok-backend, ngrok-frontend)")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    parser.add_argument("--tail", type=int, help="Number of lines to show from end")
    parser.add_argument("--no-ngrok", action="store_true", help="Exclude ngrok services (ngrok is included by default)")

    args = parser.parse_args()

    # Base command
    cmd = ["docker", "compose"]

    # Add ngrok profile by default (unless --no-ngrok is specified)
    if not args.no_ngrok:
        cmd.extend(["--profile", "ngrok"])

    cmd.append("logs")

    if args.follow:
        cmd.append("-f")

    if args.tail:
        cmd.extend(["--tail", str(args.tail)])

    if args.service:
        cmd.append(args.service)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
