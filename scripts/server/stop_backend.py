"""Enhanced script to stop the backend server"""
import psutil
import os
import sys
import time
import subprocess
from pathlib import Path

def check_port_in_use(port):
    """Check if a port is in use"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == 'LISTEN':
            return True, conn.pid
    return False, None

def clear_python_cache():
    """Clear Python __pycache__ directories"""
    try:
        # Path(__file__).parent = scripts/server/, .parent.parent = scripts/, .parent.parent.parent = project root
        project_root = Path(__file__).parent.parent.parent
        print("Clearing Python cache...")

        # Use PowerShell to remove all __pycache__ directories
        cmd = f'powershell -Command "Get-ChildItem -Path \'{project_root}\' -Filter __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force"'
        subprocess.run(cmd, shell=True, capture_output=True)
        print("[OK] Python cache cleared")
    except Exception as e:
        print(f"[WARN] Could not clear cache: {e}")

def stop_backend(force=False):
    """
    Stop the backend server running on port 8001

    Args:
        force: If True, kill ALL Python processes (use with caution)
    """
    print("=" * 60)
    print("Stopping Backend Server")
    print("=" * 60)

    found_processes = []

    # Method 1: Find by port
    print("\n[1/3] Checking port 8001...")
    port_in_use, port_pid = check_port_in_use(8001)
    if port_in_use:
        print(f"[OK] Found process using port 8001 (PID: {port_pid})")
        found_processes.append(port_pid)
    else:
        print("  No process found using port 8001")

    # Method 2: Find by uvicorn command
    print("\n[2/3] Searching for uvicorn processes...")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('uvicorn' in str(arg) for arg in cmdline):
                if any('8001' in str(arg) or 'main:app' in str(arg) for arg in cmdline):
                    if proc.info['pid'] not in found_processes:
                        print(f"[OK] Found uvicorn process (PID: {proc.info['pid']})")
                        found_processes.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not found_processes:
        print("  No uvicorn processes found")

    # Method 3: Force kill ALL Python processes (if requested)
    if force:
        print("\n[WARNING] Force mode enabled - killing ALL Python processes...")
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info.get('name', '').lower().startswith('python'):
                    if proc.info['pid'] not in found_processes:
                        print(f"[OK] Found Python process (PID: {proc.info['pid']})")
                        found_processes.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # Kill all found processes
    if found_processes:
        print(f"\n[3/3] Terminating {len(found_processes)} process(es)...")
        killed = 0
        for pid in found_processes:
            try:
                proc = psutil.Process(pid)

                # Get children first
                children = proc.children(recursive=True)

                # Terminate parent
                proc.terminate()

                # Terminate children
                for child in children:
                    try:
                        child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Wait for graceful termination
                gone, alive = psutil.wait_procs([proc] + children, timeout=3)

                # Force kill if still alive
                for p in alive:
                    try:
                        p.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                print(f"  [OK] Killed PID {pid}")
                killed += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"  [WARN] Could not kill PID {pid}: {e}")

        # Verify port is freed
        time.sleep(1)
        port_in_use, _ = check_port_in_use(8001)
        if not port_in_use:
            print("\n[OK] Port 8001 is now free")
        else:
            print("\n[WARN] Port 8001 is still in use!")
            return 1

        # Clear Python cache
        clear_python_cache()

        print("\n" + "=" * 60)
        print(f"[OK] Backend server stopped successfully! ({killed} process(es) killed)")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("[WARN] No backend server found running")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    force = '--force' in sys.argv
    if force:
        response = input("Force mode will kill ALL Python processes. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled")
            sys.exit(0)

    sys.exit(stop_backend(force=force))
