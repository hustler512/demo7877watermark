#!/usr/bin/env bash
set -euo pipefail

echo "Checking ffmpeg..."
command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }

echo "Checking python modules..."
python -c "import cv2, numpy, pytesseract" || { echo "Python deps missing"; exit 2; }

echo "Smoke checks passed"