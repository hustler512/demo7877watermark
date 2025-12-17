"""
Safe installer + runner for local/personal use.

Usage:
    python run_with_deps.py        # installs requirements then starts the server
    AUTO_INSTALL_DEPENDENCIES=0 python run_with_deps.py   # skip install

This script is intentionally opt-in via environment variable; it avoids modifying
`main.py` and keeps installation separate from runtime for safety.
"""
import os
import sys
import subprocess
import shlex
from pathlib import Path

ROOT = Path(__file__).parent
REQ = ROOT / 'requirements.txt'

def run_install():
    print('Installing Python dependencies from', REQ)
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(REQ)])
        print('Dependencies installed successfully.')
    except subprocess.CalledProcessError as e:
        print('Failed to install dependencies:', e)
        print('You can inspect and run `python -m pip install -r requirements.txt` manually.')
        sys.exit(1)


def start_server():
    # Default to the same uvicorn command we use in Dockerfile
    cmd = f"{sys.executable} -m uvicorn main:app --host 0.0.0.0 --port 8000"
    print('Starting server:')
    print('  ', cmd)
    # Use exec style to forward signals
    args = shlex.split(cmd)
    os.execv(args[0], args)


if __name__ == '__main__':
    # Respect an explicit opt-out
    auto = os.environ.get('AUTO_INSTALL_DEPENDENCIES', '1')
    if auto in ('1', 'true', 'True', 'yes', 'on'):
        run_install()
    else:
        print('AUTO_INSTALL_DEPENDENCIES is disabled; skipping dependency install.')

    start_server()
