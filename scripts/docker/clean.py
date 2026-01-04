#!/usr/bin/env python3
"""Clean up Docker resources"""
import subprocess
import sys
import argparse

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Clean up Docker resources")
    parser.add_argument("--all", action="store_true",
                       help="Remove images and volumes (WARNING: deletes everything)")
    parser.add_argument("--images", action="store_true", help="Remove images only")
    parser.add_argument("--volumes", action="store_true", help="Remove volumes only")
    parser.add_argument("--ngrok", action="store_true", help="Include ngrok services in cleanup")

    args = parser.parse_args()

    # Base command
    base_cmd = ["docker", "compose"]
    if args.ngrok:
        base_cmd.extend(["--profile", "ngrok"])

    if args.all:
        print("WARNING: This will remove all containers, images, and volumes!")
        print("All data will be deleted!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            sys.exit(0)

        # Stop and remove containers and volumes
        subprocess.run(base_cmd + ["down", "-v"])

        # Remove images
        subprocess.run(["docker", "rmi", "chatterbox-backend", "chatterbox-frontend"],
                      stderr=subprocess.DEVNULL)

        print("[OK] Cleanup complete!")

    elif args.images:
        print("Removing Docker images...")
        subprocess.run(["docker", "rmi", "chatterbox-backend", "chatterbox-frontend"],
                      stderr=subprocess.DEVNULL)
        print("[OK] Images removed!")

    elif args.volumes:
        print("WARNING: This will remove all volumes and delete data!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            sys.exit(0)

        subprocess.run(base_cmd + ["down", "-v"])
        print("[OK] Volumes removed!")

    else:
        # Just stop and remove containers (default)
        subprocess.run(base_cmd + ["down"])
        print("[OK] Containers stopped and removed!")


if __name__ == "__main__":
    main()
