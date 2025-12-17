import os
from video_processor import VideoProcessor


def test_processor_import():
    # basic import smoke test
    assert hasattr(VideoProcessor, 'extract_frames')


def test_dummy_run(tmp_path):
    # create a tiny dummy frame to simulate processing flow (no ffmpeg)
    inp = tmp_path / 'input.mp4'
    out = tmp_path / 'out.mp4'
    # touch files to avoid errors
    inp.write_text('')
    vp = VideoProcessor(str(inp), str(out), job_id='smoketest')
    # ensure dirs created
    assert vp.frames_dir.exists()
    vp.work_dir.mkdir(parents=True, exist_ok=True)
    # cleanup
    vp.cleanup()
