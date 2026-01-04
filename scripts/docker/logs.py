#!/usr/bin/env python3
"""View Docker container logs"""
import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="View Docker container logs")
    parser.add_argument("service", nargs="?", help="Service name (backend, frontend, postgres)")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    parser.add_argument("--tail", type=int, help="Number of lines to show from end")

    args = parser.parse_args()

    cmd = ["docker-compose", "logs"]

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
