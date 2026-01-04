"""Enhanced script to stop both backend and frontend servers"""
import sys
import subprocess

def stop_all(force=False):
    """
    Stop both backend and frontend servers

    Args:
        force: If True, kill ALL Python and Node processes
    """
    print("=" * 60)
    print("Stopping All Servers")
    print("=" * 60)

    args = ['--force'] if force else []

    # Stop backend
    print("\n1. Stopping Backend Server:")
    backend_result = subprocess.run([sys.executable, "scripts/server/stop_backend.py"] + args)

    # Stop frontend
    print("\n2. Stopping Frontend Server:")
    frontend_result = subprocess.run([sys.executable, "scripts/server/stop_frontend.py"] + args)

    print("\n" + "=" * 60)
    if backend_result.returncode == 0 or frontend_result.returncode == 0:
        print("[OK] Server shutdown complete!")
    else:
        print("[WARN] No servers were running")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    force = '--force' in sys.argv
    if force:
        response = input("Force mode will kill ALL Python and Node processes. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled")
            sys.exit(0)

    sys.exit(stop_all(force=force))
