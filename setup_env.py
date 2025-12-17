import subprocess
import sys
import os

def install_requirements(requirements='requirements.txt'):
    base = os.path.dirname(__file__)
    req_path = os.path.join(base, requirements)
    if not os.path.exists(req_path):
        req_path = requirements

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_path])

if __name__ == '__main__':
    try:
        install_requirements()
        print('Dependencies installed')
    except subprocess.CalledProcessError as e:
        print('Failed to install dependencies:', e)
        sys.exit(1)
