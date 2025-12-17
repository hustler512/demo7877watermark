@echo off
REM Install Python dependencies from project requirements.txt
python -m pip install --upgrade pip
python -m pip install -r "%~dp0..\requirements.txt"
echo Dependencies installed.
pause
