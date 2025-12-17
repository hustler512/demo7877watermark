import os
import shutil
import threading
import subprocess
from pathlib import Path
from video_processor import VideoProcessor

jobs = {}  # workers will update the same job structure as main


def cleanup_files_later(job_id: str, delay: int = 3600):
    import time
    try:
        time.sleep(delay)
        j = jobs.get(job_id)
        if not j:
            return
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
        tempdir = Path('temp') / job_id
        if tempdir.exists():
            try:
                shutil.rmtree(tempdir)
            except Exception:
                pass
        try:
            jobs.pop(job_id, None)
        except Exception:
            pass
    except Exception:
        pass


def process_video_task(job_id, input_path, output_path, task, mask_coords, remove_audio, auto_detect):
    try:
        # Note: this function mirrors the processing logic used by main.py
        jobs[job_id]['progress'] = 20
        proc = VideoProcessor(input_path, output_path, job_id)

        jobs[job_id]['progress'] = 30
        proc.extract_frames()

        frame_files = sorted(proc.frames_dir.glob('*.png'))
        total_frames = len(frame_files)

        jobs[job_id]['progress'] = 40
        audio = proc.extract_audio()

        jobs[job_id]['progress'] = 50
        if task == 'caption' and not auto_detect:
            no_subs = str(Path(output_path).with_suffix('.nosubs' + Path(output_path).suffix))
            try:
                proc.remove_soft_subtitles(no_subs)
                shutil.move(no_subs, output_path)
                jobs[job_id]['progress'] = 100
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['output'] = output_path
                threading.Thread(target=cleanup_files_later, args=(job_id,)).start()
                return
            except Exception:
                pass

        jobs[job_id]['progress'] = 55
        if task in ('watermark', 'caption'):
            if task == 'watermark' and auto_detect and not mask_coords:
                try:
                    detected = proc.detect_watermark()
                    if detected:
                        mask_coords = detected
                        jobs[job_id]['progress'] = 58
                except Exception:
                    pass
            if task == 'caption' and auto_detect:
                boxes, combined = proc.detect_captions()
                if combined:
                    mask_coords = combined

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
        use_nvenc = os.environ.get('USE_NVENC', '0') == '1'
        proc.reconstruct_video(remove_audio=remove_audio, use_nvenc=use_nvenc)

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['output'] = output_path

        threading.Thread(target=cleanup_files_later, args=(job_id,)).start()
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        print('Processing error:', e)
