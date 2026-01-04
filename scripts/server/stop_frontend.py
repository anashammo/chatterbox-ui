"""Enhanced script to stop the Angular frontend server"""
import psutil
import os
import sys
import time

def check_port_in_use(port):
    """Check if a port is in use"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == 'LISTEN':
            return True, conn.pid
    return False, None

def stop_frontend(force=False):
    """
    Stop the Angular dev server running on port 4200

    Args:
        force: If True, kill ALL Node processes (use with caution)
    """
    print("=" * 60)
    print("Stopping Frontend Server")
    print("=" * 60)

    found_processes = []

    # Method 1: Find by port
    print("\n[1/3] Checking port 4200...")
    port_in_use, port_pid = check_port_in_use(4200)
    if port_in_use:
        print(f"[OK] Found process using port 4200 (PID: {port_pid})")
        found_processes.append(port_pid)
    else:
        print("  No process found using port 4200")

    # Method 2: Find by Node + ng command
    print("\n[2/3] Searching for Node/ng serve processes...")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            name = proc.info.get('name', '').lower()

            # Check if it's a Node process
            if name.startswith('node') and cmdline:
                # Check if it's related to Angular (ng serve) or port 4200
                if any('ng' in str(arg) or '4200' in str(arg) or 'angular' in str(arg).lower() for arg in cmdline):
                    if proc.info['pid'] not in found_processes:
                        print(f"[OK] Found ng serve process (PID: {proc.info['pid']})")
                        found_processes.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not found_processes:
        print("  No ng serve processes found")

    # Method 3: Force kill ALL Node processes (if requested)
    if force:
        print("\n[WARNING] Force mode enabled - killing ALL Node processes...")
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info.get('name', '').lower()
                if name.startswith('node'):
                    if proc.info['pid'] not in found_processes:
                        print(f"[OK] Found Node process (PID: {proc.info['pid']})")
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

                # Get children first (Angular spawns many child processes)
                children = proc.children(recursive=True)

                # Terminate parent
                proc.terminate()

                # Terminate all children
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

                print(f"  [OK] Killed PID {pid} and {len(children)} child process(es)")
                killed += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"  [WARN] Could not kill PID {pid}: {e}")

        # Verify port is freed
        time.sleep(1)
        port_in_use, _ = check_port_in_use(4200)
        if not port_in_use:
            print("\n[OK] Port 4200 is now free")
        else:
            print("\n[WARN] Port 4200 is still in use!")
            return 1

        print("\n" + "=" * 60)
        print(f"[OK] Frontend server stopped successfully! ({killed} process(es) killed)")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("[WARN] No frontend server found running")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    force = '--force' in sys.argv
    if force:
        response = input("Force mode will kill ALL Node processes. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled")
            sys.exit(0)

    sys.exit(stop_frontend(force=force))
