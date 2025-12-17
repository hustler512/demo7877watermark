import shutil
import uuid
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import subprocess
from video_processor import VideoProcessor

app = FastAPI(title="CleanVideo Pro API")

# Directories
BASE = Path('.')
UPLOAD_DIR = BASE / 'uploads'
PROCESSED_DIR = BASE / 'processed'
TEMP_DIR = BASE / 'temp'
for d in [UPLOAD_DIR, PROCESSED_DIR, TEMP_DIR]:
    d.mkdir(exist_ok=True)

# Simple in-memory job store
jobs = {}

# Static files and templates
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/upload')
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        return JSONResponse(status_code=400, content={'error': 'Unsupported file type'})

    # size check (500MB)
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > 500 * 1024 * 1024:
        return JSONResponse(status_code=400, content={'error': 'File too large (limit 500MB)'})

    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    out_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(out_path, 'wb') as buf:
        shutil.copyfileobj(file.file, buf)

    # duration check
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(r.stdout.strip()) if r.stdout.strip() else 0.0
    except Exception:
        duration = 0.0

    if duration > 300 and duration > 0.0:
        out_path.unlink()
        return JSONResponse(status_code=400, content={'error': 'Video duration too long (max 5 minutes for MVP)'})

    jobs[job_id] = {
        'status': 'uploaded',
        'file': str(out_path),
        'progress': 0,
        'filename': file.filename
    }

    return {'job_id': job_id, 'filename': file.filename, 'duration': duration}

@app.post('/process')
async def process_video(background_tasks: BackgroundTasks,
                        job_id: str = Form(...),
                        task: str = Form(...),  # 'watermark' or 'caption' or 'audio'
                        mask_x: int = Form(None),
                        mask_y: int = Form(None),
                        mask_w: int = Form(None),
                        mask_h: int = Form(None),
                        remove_audio: bool = Form(False),
                        auto_detect: bool = Form(True)):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={'error': 'Job not found'})

    job = jobs[job_id]
    if job['status'] == 'processing':
        return JSONResponse(status_code=400, content={'error': 'Job already processing'})

    job['status'] = 'processing'
    job['progress'] = 10

    input_path = job['file']
    ext = Path(input_path).suffix
    output_path = PROCESSED_DIR / f"{job_id}_clean{ext}"

    mask_coords = None
    if mask_x is not None and mask_y is not None and mask_w is not None and mask_h is not None:
        mask_coords = (mask_x, mask_y, mask_w, mask_h)

    # If REDIS_URL set, enqueue with RQ for scaling; otherwise run in background task
    redis_url = os.environ.get('REDIS_URL')
    if redis_url:
        try:
            import redis
            from rq import Queue
            conn = redis.from_url(redis_url)
            q = Queue(connection=conn)
            q.enqueue(process_video_task, job_id, input_path, str(output_path), task, mask_coords, remove_audio, auto_detect)
            jobs[job_id]['status'] = 'enqueued'
            return {'job_id': job_id, 'status': 'enqueued'}
        except Exception as e:
            # fallback to local background processing
            jobs[job_id]['status'] = 'processing'
            background_tasks.add_task(process_video_task, job_id, input_path, str(output_path), task, mask_coords, remove_audio, auto_detect)
            return {'job_id': job_id, 'status': 'processing', 'note': 'Redis enqueue failed, processing locally', 'error': str(e)}

    background_tasks.add_task(process_video_task, job_id, input_path, str(output_path), task, mask_coords, remove_audio, auto_detect)

    return {'job_id': job_id, 'status': 'processing'}


def cleanup_files_later(job_id: str, delay: int = 3600):
    """Background helper to remove job files after `delay` seconds."""
    import time
    try:
        time.sleep(delay)
        j = jobs.get(job_id)
        if not j:
            return
        # delete uploaded and output
        if 'file' in j and os.path.exists(j['file']):
            try:
                os.remove(j['file'])
            except Exception:
                pass
        out = j.get('output')
        if out and os.path.exists(out):
            try:
                os.remove(out)
            except Exception:
                pass
        # cleanup temp
        tempdir = Path('temp') / job_id
        if tempdir.exists():
            try:
                shutil.rmtree(tempdir)
            except Exception:
                pass
        # finally remove job entry
        try:
            jobs.pop(job_id, None)
        except Exception:
            pass
    except Exception:
        pass


def process_video_task(job_id, input_path, output_path, task, mask_coords, remove_audio, auto_detect):
    try:
        jobs[job_id]['progress'] = 20
        proc = VideoProcessor(input_path, output_path, job_id)

        jobs[job_id]['progress'] = 30
        proc.extract_frames()

        # Count frames for progress reporting
        frame_files = sorted(proc.frames_dir.glob('*.png'))
        total_frames = len(frame_files)

        jobs[job_id]['progress'] = 40
        audio = proc.extract_audio()

        jobs[job_id]['progress'] = 50
        # Try soft-subtitle removal if requested explicitly
        if task == 'caption' and not auto_detect:
            no_subs = str(Path(output_path).with_suffix('.nosubs' + Path(output_path).suffix))
            try:
                proc.remove_soft_subtitles(no_subs)
                shutil.move(no_subs, output_path)
                jobs[job_id]['progress'] = 100
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['output'] = output_path
                # schedule cleanup
                import threading
                threading.Thread(target=cleanup_files_later, args=(job_id,)).start()
                return
            except Exception:
                pass

        jobs[job_id]['progress'] = 55
        if task in ('watermark', 'caption'):
            # For watermark task, if auto-detect requested and no manual mask, try watermark detection
            if task == 'watermark' and auto_detect and not mask_coords:
                try:
                    detected = proc.detect_watermark()
                    if detected:
                        mask_coords = detected
                        jobs[job_id]['progress'] = 58
                except Exception:
                    pass
            # If caption auto-detect requested, try to detect boxes and use combined bbox
            if task == 'caption' and auto_detect:
                boxes, combined = proc.detect_captions()
                if combined:
                    mask_coords = combined

            # progress range for frames: 55 -> 80
            def frame_progress_callback(done, total):
                try:
                    base = 55
                    end = 80
                    pct = base + int((done/total) * (end-base))
                    jobs[job_id]['progress'] = pct
                except Exception:
                    pass

            max_workers = int(os.environ.get('MAX_WORKERS', 0)) or None
            proc.process_frames(mask_coords=mask_coords, use_auto=auto_detect, progress_callback=frame_progress_callback, max_workers=max_workers)

        jobs[job_id]['progress'] = 80
        # optionally enable nvenc via env var
        use_nvenc = os.environ.get('USE_NVENC', '0') == '1'
        proc.reconstruct_video(remove_audio=remove_audio, use_nvenc=use_nvenc)

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['output'] = output_path

        # schedule cleanup after 1 hour in background thread
        import threading
        threading.Thread(target=cleanup_files_later, args=(job_id,)).start()
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        print('Processing error:', e)

@app.get('/status/{job_id}')
async def status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={'error': 'Job not found'})
    job = jobs[job_id]
    return {'job_id': job_id, 'status': job['status'], 'progress': job.get('progress', 0), 'error': job.get('error')}


@app.get('/preview/{job_id}')
async def preview(job_id: str, frame_index: int = None, mask_x: int = None, mask_y: int = None, mask_w: int = None, mask_h: int = None, task: str = 'watermark', use_auto: bool = True):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={'error': 'Job not found'})
    job = jobs[job_id]
    input_path = job.get('file')
    if not input_path or not os.path.exists(input_path):
        return JSONResponse(status_code=404, content={'error': 'Source file missing'})

    try:
        proc = VideoProcessor(input_path, str(PROCESSED_DIR / f"{job_id}_clean.mp4"), job_id)
        mask_coords = None
        if mask_x is not None and mask_y is not None and mask_w is not None and mask_h is not None:
            mask_coords = (mask_x, mask_y, mask_w, mask_h)

        result = proc.preview_inpaint(frame_index=frame_index, mask_coords=mask_coords, use_auto=use_auto, task=task)
        return FileResponse(result.path, media_type='image/png')
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.get('/download/{job_id}')
async def download(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={'error': 'Job not found'})
    job = jobs[job_id]
    if job['status'] != 'completed':
        return JSONResponse(status_code=400, content={'error': 'Not ready'})
    path = job.get('output')
    if not path or not os.path.exists(path):
        return JSONResponse(status_code=404, content={'error': 'File missing'})
    return FileResponse(path, media_type='video/mp4', filename=f"clean_{job['filename']}")

@app.delete('/cancel/{job_id}')
async def cancel(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={'error': 'Job not found'})

    # best-effort cleanup
    j = jobs.pop(job_id)
    # remove uploaded and processed files
    try:
        if 'file' in j and os.path.exists(j['file']):
            os.remove(j['file'])
        out = j.get('output')
        if out and os.path.exists(out):
            os.remove(out)
    except Exception:
        pass
    return {'message': 'cancelled'}

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
