"""Script to start the Chatterbox TTS Angular frontend server"""
import subprocess
import sys
import os

def run_frontend():
    """Start the Angular dev server on port 4201"""
    print("=" * 50)
    print("Starting Chatterbox TTS Frontend Server...")
    print("=" * 50)

    # Change to frontend directory
    # Get project root: scripts/server/ -> scripts/ -> project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    frontend_dir = os.path.join(
        project_root,
        "src", "presentation", "frontend"
    )

    if not os.path.exists(frontend_dir):
        print(f"Error: Frontend directory not found: {frontend_dir}")
        return 1

    print(f"\nFrontend directory: {frontend_dir}")
    print("Running: ng serve --port 4201")
    print("Frontend URL: http://localhost:4201")
    print("-" * 50)

    try:
        # Run ng serve with custom port in the frontend directory
        result = subprocess.run(
            ["npx", "ng", "serve", "--port", "4201"],
            cwd=frontend_dir,
            shell=True
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nFrontend server stopped by user")
        return 0
    except Exception as e:
        print(f"\nError starting frontend: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_frontend())
