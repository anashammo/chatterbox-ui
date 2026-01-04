"""
Smart Commit Helper - Interactive commit creator with standardized messages

Creates consistent commit messages following the project's conventions.
Auto-appends standard footer with Claude Code attribution.

Usage:
    python scripts/git/smart_commit.py
    python scripts/git/smart_commit.py --type feat --scope ui --message "Add new feature"
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.terminal import Colors


COMMIT_TYPES = {
    'feat': 'Feat: New feature',
    'fix': 'Fix: Bug fix',
    'refactor': 'Refactor: Code improvement without behavior change',
    'docs': 'Docs: Documentation update',
    'style': 'Style: Code formatting/style changes',
    'test': 'Test: Add or update tests',
    'chore': 'Chore: Maintenance task',
    'perf': 'Perf: Performance improvement',
}

STANDARD_FOOTER = """
🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
"""


def run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a shell command and return status, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def get_git_status() -> Tuple[List[str], List[str]]:
    """Get staged and unstaged changes"""
    # Staged files
    returncode, stdout, _ = run_command(['git', 'diff', '--cached', '--name-only'])
    staged = stdout.strip().split('\n') if returncode == 0 and stdout.strip() else []

    # Unstaged files
    returncode, stdout, _ = run_command(['git', 'diff', '--name-only'])
    unstaged = stdout.strip().split('\n') if returncode == 0 and stdout.strip() else []

    return staged, unstaged


def get_current_branch() -> str:
    """Get current git branch name"""
    returncode, stdout, _ = run_command(['git', 'branch', '--show-current'])
    return stdout.strip() if returncode == 0 else "unknown"


def suggest_commit_type(files: List[str]) -> str:
    """Suggest commit type based on changed files"""
    if not files or files == ['']:
        return 'feat'

    # Count file types
    has_docs = any('.md' in f or 'README' in f for f in files)
    has_tests = any('test' in f.lower() or 'spec' in f.lower() for f in files)
    has_frontend = any('frontend' in f or '.html' in f or '.css' in f or '.ts' in f for f in files)
    has_backend = any('backend' in f or 'api' in f or '.py' in f for f in files)

    if has_docs:
        return 'docs'
    elif has_tests:
        return 'test'
    elif has_frontend or has_backend:
        return 'feat'
    else:
        return 'feat'


def suggest_scope(files: List[str]) -> str:
    """Suggest commit scope based on changed files"""
    if not files or files == ['']:
        return ''

    # Analyze file paths
    has_frontend = any('frontend' in f for f in files)
    has_backend = any(any(x in f for x in ['api', 'routers', 'use_cases', 'domain', 'infrastructure']) for f in files)
    has_docs = any('.md' in f for f in files)
    has_scripts = any('scripts' in f for f in files)

    if has_frontend and has_backend:
        return 'fullstack'
    elif has_frontend:
        return 'frontend'
    elif has_backend:
        return 'backend'
    elif has_docs:
        return 'docs'
    elif has_scripts:
        return 'scripts'
    else:
        return ''


def format_commit_message(commit_type: str, scope: str, message: str, body: str = "") -> str:
    """Format commit message with standard footer"""
    # Build the commit message
    type_prefix = commit_type.capitalize()

    if scope:
        first_line = f"{type_prefix}({scope}): {message}"
    else:
        first_line = f"{type_prefix}: {message}"

    full_message = first_line

    if body:
        full_message += f"\n\n{body}"

    full_message += STANDARD_FOOTER

    return full_message


def interactive_commit():
    """Interactive commit creation"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}Smart Commit Helper{Colors.RESET}\n")

    # Check git status
    staged, unstaged = get_git_status()
    branch = get_current_branch()

    print(f"{Colors.BLUE}Current branch:{Colors.RESET} {branch}")

    if not staged or staged == ['']:
        print(f"\n{Colors.RED}No staged changes found!{Colors.RESET}")
        print(f"{Colors.YELLOW}Run 'git add <files>' to stage changes first.{Colors.RESET}\n")

        if unstaged and unstaged != ['']:
            print(f"{Colors.YELLOW}Unstaged files:{Colors.RESET}")
            for f in unstaged[:10]:  # Show max 10 files
                print(f"  - {f}")
            if len(unstaged) > 10:
                print(f"  ... and {len(unstaged) - 10} more")

        return False

    print(f"\n{Colors.GREEN}Staged files ({len(staged)}):{Colors.RESET}")
    for f in staged[:15]:  # Show max 15 files
        print(f"  - {f}")
    if len(staged) > 15:
        print(f"  ... and {len(staged) - 15} more")

    # Suggest type and scope
    suggested_type = suggest_commit_type(staged)
    suggested_scope = suggest_scope(staged)

    # Select commit type
    print(f"\n{Colors.CYAN}Select commit type:{Colors.RESET}")
    for i, (key, desc) in enumerate(COMMIT_TYPES.items(), 1):
        marker = f"{Colors.GREEN}>{Colors.RESET}" if key == suggested_type else " "
        print(f"  {marker} [{i}] {key}: {desc}")

    type_input = input(f"\n{Colors.BLUE}Enter number or type name (default: {suggested_type}):{Colors.RESET} ").strip()

    if type_input.isdigit():
        type_index = int(type_input) - 1
        if 0 <= type_index < len(COMMIT_TYPES):
            commit_type = list(COMMIT_TYPES.keys())[type_index]
        else:
            commit_type = suggested_type
    elif type_input in COMMIT_TYPES:
        commit_type = type_input
    else:
        commit_type = suggested_type

    # Enter scope
    scope_input = input(f"{Colors.BLUE}Enter scope (default: {suggested_scope}):{Colors.RESET} ").strip()
    scope = scope_input if scope_input else suggested_scope

    # Enter message
    while True:
        message = input(f"{Colors.BLUE}Enter commit message:{Colors.RESET} ").strip()
        if message:
            break
        print(f"{Colors.RED}Message cannot be empty!{Colors.RESET}")

    # Optional body
    print(f"\n{Colors.YELLOW}Enter commit body (optional, press Enter to skip):{Colors.RESET}")
    print(f"{Colors.YELLOW}Type your message and press Ctrl+Z (Windows) or Ctrl+D (Unix) when done:{Colors.RESET}")
    try:
        body_lines = []
        while True:
            try:
                line = input()
                body_lines.append(line)
            except EOFError:
                break
        body = '\n'.join(body_lines).strip()
    except KeyboardInterrupt:
        body = ""

    # Format the commit message
    full_message = format_commit_message(commit_type, scope, message, body)

    # Preview
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}Commit Preview:{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(full_message)
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

    # Confirm
    confirm = input(f"{Colors.GREEN}Create this commit? (Y/n):{Colors.RESET} ").strip().lower()

    if confirm in ['', 'y', 'yes']:
        # Create the commit
        returncode, _, stderr = run_command(['git', 'commit', '-m', full_message])

        if returncode == 0:
            print(f"\n{Colors.GREEN}[OK] Commit created successfully!{Colors.RESET}")

            # Show commit hash
            returncode, stdout, _ = run_command(['git', 'log', '-1', '--format=%H %s'])
            if returncode == 0:
                print(f"{Colors.BLUE}Commit:{Colors.RESET} {stdout.strip()}")

            return True
        else:
            print(f"\n{Colors.RED}[FAIL] Commit failed!{Colors.RESET}")
            print(f"{Colors.RED}{stderr}{Colors.RESET}")
            return False
    else:
        print(f"\n{Colors.YELLOW}Commit cancelled.{Colors.RESET}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart commit helper with standardized messages')
    parser.add_argument('--type', choices=COMMIT_TYPES.keys(), help='Commit type')
    parser.add_argument('--scope', help='Commit scope (e.g., frontend, backend)')
    parser.add_argument('--message', '-m', help='Commit message')
    parser.add_argument('--body', '-b', help='Commit body')
    parser.add_argument('--interactive', '-i', action='store_true', help='Force interactive mode')

    args = parser.parse_args()

    # If all required args provided, create commit directly
    if args.type and args.message and not args.interactive:
        full_message = format_commit_message(
            args.type,
            args.scope or '',
            args.message,
            args.body or ''
        )

        print(f"\n{Colors.CYAN}Creating commit...{Colors.RESET}\n")
        print(full_message)

        returncode, _, stderr = run_command(['git', 'commit', '-m', full_message])

        if returncode == 0:
            print(f"\n{Colors.GREEN}[OK] Commit created successfully!{Colors.RESET}")
            return 0
        else:
            print(f"\n{Colors.RED}[FAIL] Commit failed!{Colors.RESET}")
            print(f"{Colors.RED}{stderr}{Colors.RESET}")
            return 1
    else:
        # Interactive mode
        success = interactive_commit()
        return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}ERROR: {str(e)}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
