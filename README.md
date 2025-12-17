# demo7877watermark

Small local web app to remove watermarks and captions from videos (FastAPI + OpenCV + ffmpeg).

## Installation (Windows, local machine)

1. Open PowerShell or Command Prompt in the project folder.
2. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Ensure `ffmpeg` is installed and on PATH. For caption detection also install `tesseract-ocr`.

## Run locally (same Wi‑Fi access)

1. Start the server:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

2. On your phone (connected to the same Wi‑Fi) open http://<PC_LOCAL_IP>:8000/ — find your PC IP with `ipconfig`.

## One-command install & run (local / same Wi‑Fi)

If you want one command that installs dependencies and starts the server, use `run_with_deps.py`.

Default behavior: it will install from `requirements.txt` then start the app. To skip the automatic install, set the env var `AUTO_INSTALL_DEPENDENCIES=0`.

```bash
python run_with_deps.py
# or to skip install
AUTO_INSTALL_DEPENDENCIES=0 python run_with_deps.py
```

## Docker / Compose (recommended for reproducible local deploy)

An example `docker-compose.yml` can start the app and a Redis server for job queuing (if you enable it):

```bash
docker-compose up --build
```

If you want to run just the app in Docker locally:

```bash
docker build -t cleanvideo .
docker run -p 8000:8000 -v %cd%/uploads:/app/uploads -v %cd%/processed:/app/processed cleanvideo
```

## Notes on pushing changes (if git not found)

If `git` isn't available in your PowerShell session, add it to PATH for the current session (replace path if Git installed elsewhere):

```powershell
$env:Path += ';C:\Program Files\Git\cmd'
git --version
```

Then commit and push from the project folder:

```powershell
cd "C:\Users\pedro.muchanga\OneDrive - Meridian Group\Documents\GitHub\demo7877watermark"
git add -A
git commit -m "Add install helpers, worker tasks and AI upscale prototype"
git push
```

## Notes and safety

- The project stores uploads in `uploads`, processed files in `processed`, and temp in `temp`.
- For personal/local use this repo is runnable as an MVP. For public hosting you should add authentication, persistent job storage (Redis/S3), HTTPS, and tighten CORS.
- Automatic install helpers are for local use only; for production use containerized deploy with pinned images.
# demo7877watermark

Small local web app to remove watermarks and captions from videos (FastAPI + OpenCV + ffmpeg).

## Installation (Windows, local machine)

1. Open PowerShell or Command Prompt in the project folder.
2. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Ensure `ffmpeg` is installed and on PATH. For caption detection also install `tesseract-ocr`.

## Run locally (same Wi‑Fi access)

1. Start the server:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

2. On your phone (connected to the same Wi‑Fi) open http://<PC_LOCAL_IP>:8000/ — find your PC IP with `ipconfig`.

## Notes

- The project stores uploads in `uploads`, processed files in `processed`, and temp in `temp`.
- For personal/local use this repo is runnable as an MVP. For public hosting you should add authentication, persistent job storage (Redis/S3), HTTPS, and tighten CORS.
- Automated install helper: see `scripts/install_requirements.bat` and `scripts/install_requirements.ps1`, and `setup_env.py` for a programmatic installer.

## One-command install & run (local / same Wi‑Fi)

If you want one command that installs dependencies and starts the server, use `run_with_deps.py`.

Default behavior: it will install from `requirements.txt` then start the app. To skip the automatic install, set the env var `AUTO_INSTALL_DEPENDENCIES=0`.

```bash
python run_with_deps.py
# or to skip install
AUTO_INSTALL_DEPENDENCIES=0 python run_with_deps.py
```

Note: automatic installation is intended for simple local/personal use. For production or shared hosts, prefer manual dependency management and containerized deployment.
# demo7877watermark