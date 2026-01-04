#!/usr/bin/env python3
"""Open shell in running container"""
import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Open shell in running container")
    parser.add_argument("service", choices=["backend", "frontend", "postgres"],
                       help="Service to open shell in")

    args = parser.parse_args()

    # Determine container name and shell
    container_map = {
        "backend": ("chatterbox-backend", "bash"),
        "frontend": ("chatterbox-frontend", "sh"),
        "postgres": ("chatterbox-postgres", "sh")
    }

    container_name, shell = container_map[args.service]

    cmd = ["docker", "exec", "-it", container_name, shell]

    print(f"Opening {shell} in {container_name}...")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
