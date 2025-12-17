import os
import subprocess
import shutil
from pathlib import Path
import cv2
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _process_single_frame(args):
    """Worker function for parallel processing. Runs in a separate process."""
    frame_path, out_dir, mask_coords, use_auto = args
    try:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return str(frame_path), False

        # Build mask
        if mask_coords:
            x, y, w, h = mask_coords
            mm = np.zeros(frame.shape[:2], dtype=np.uint8)
            mm[y:y+h, x:x+w] = 255
            mask = mm
        elif use_auto:
            # Simple auto-detect performed per-frame
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            mask = np.zeros((h, w), dtype=np.uint8)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x2, y2, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                if area < 100:
                    continue
                if (x2 < w * 0.2 and y2 < h * 0.2) or (x2 + cw > w * 0.8 and y2 + ch > h * 0.8) or (y2 + ch > h * 0.8):
                    cv2.rectangle(mask, (x2, y2), (x2 + cw, y2 + ch), 255, -1)
        else:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Attempt to use optional LaMa model (if available in this worker); otherwise use OpenCV
        try:
            # lazy-load per-process model if present
            model_path = Path('models') / 'lama.pt'
            if model_path.exists():
                try:
                    import torch
                    # load model as torch.jit if possible (user-provided)
                    m = torch.jit.load(str(model_path))
                    # prepare tensors
                    ft = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                    with torch.no_grad():
                        out = m(ft, mt)
                    out_img = (out.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                    inpainted = out_img
                except Exception:
                    inpainted = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            else:
                inpainted = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        except Exception:
            inpainted = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        out_path = Path(out_dir) / Path(frame_path).name
        cv2.imwrite(str(out_path), inpainted)
        return str(frame_path), True
    except Exception as e:
        return str(frame_path), False


def _inpaint_with_optional_lama(frame: np.ndarray, mask: np.ndarray):
    """Helper to inpaint using LaMa if a model file exists, otherwise OpenCV."""
    try:
        model_path = Path('models') / 'lama.pt'
        if model_path.exists():
            import torch
            m = torch.jit.load(str(model_path))
            ft = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                out = m(ft, mt)
            out_img = (out.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            return out_img
    except Exception:
        pass
    return cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def _nms_boxes(boxes, iou_threshold=0.4):
    """Simple NMS for boxes in format (x,y,w,h)."""
    if not boxes:
        return []
    # convert to x1,y1,x2,y2
    arr = []
    for b in boxes:
        x1 = b[0]
        y1 = b[1]
        x2 = b[0] + b[2]
        y2 = b[1] + b[3]
        arr.append([x1, y1, x2, y2])
    arr = np.array(arr)
    x1 = arr[:,0]
    y1 = arr[:,1]
    x2 = arr[:,2]
    y2 = arr[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    out = [boxes[i] for i in keep]
    return out


class PreviewResult:
    def __init__(self, path: str):
        self.path = path


class VideoProcessor:
    """Simple video processor: extracts frames, applies inpainting to mask areas,
    reconstructs video, and optionally strips audio or subtitle tracks.

    Uses local FFmpeg and OpenCV only (no external APIs).
    """

    def __init__(self, input_path: str, output_path: str, job_id: str, fps: int = 30):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.job_id = job_id
        self.fps = fps

        self.work_dir = Path('temp') / job_id
        self.frames_dir = self.work_dir / 'frames'
        self.processed_dir = self.work_dir / 'processed'
        self.audio_path = self.work_dir / 'audio.aac'

        self._ensure_dirs()

    def _ensure_dirs(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def get_duration(self):
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(self.input_path)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    def extract_frames(self):
        # Use ffmpeg to extract frames at target fps - allow multi-threaded ffmpeg
        pattern = str(self.frames_dir / 'frame_%06d.png')
        cmd = [
            'ffmpeg', '-y', '-i', str(self.input_path), '-vf', f'fps={self.fps}', pattern
        ]
        subprocess.run(cmd, check=True)

    def extract_audio(self):
        # Copy audio (codec copy) if exists
        cmd = ['ffmpeg', '-y', '-i', str(self.input_path), '-vn', '-acodec', 'copy', str(self.audio_path)]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self.audio_path if self.audio_path.exists() else None

    def detect_captions(self, max_bottom_ratio: float = 0.4):
        """Detect text/caption regions using pytesseract on a representative frame.

        Returns: list of boxes [(x,y,w,h), ...] and a combined bounding box (x,y,w,h) if any found.
        """
        try:
            from PIL import Image
            import pytesseract
        except Exception:
            return [], None

        # pick a middle frame for detection
        frame_files = sorted(self.frames_dir.glob('*.png'))
        if not frame_files:
            return [], None

        idx = len(frame_files) // 2
        img = cv2.imread(str(frame_files[idx]))
        if img is None:
            return [], None

        h, w = img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
        boxes = []
        for i, text in enumerate(data.get('text', [])):
            if not text or not text.strip():
                continue
            x = int(data['left'][i])
            y = int(data['top'][i])
            ww = int(data['width'][i])
            hh = int(data['height'][i])
            # Only consider text in bottom portion of frame (likely captions)
            if y > h * (1 - max_bottom_ratio):
                boxes.append((x, y, ww, hh))

        if not boxes:
            return [], None

        # compute combined bbox
        xs = [b[0] for b in boxes]
        ys = [b[1] for b in boxes]
        x2s = [b[0] + b[2] for b in boxes]
        y2s = [b[1] + b[3] for b in boxes]
        bx = min(xs)
        by = min(ys)
        bw = max(x2s) - bx
        bh = max(y2s) - by
        return boxes, (bx, by, bw, bh)

    def detect_watermark(self):
        """Attempt to detect a watermark/logo region.

        Strategy:
        - If a YOLO TorchScript model exists at `models/yolo.pt`, use it to detect logos.
        - Else if an EAST model exists at `models/frozen_east_text_detection.pb`, use it to detect text-like regions (helpful for text watermarks).
        - Fallback: use simple edge-based corner heuristic.

        Returns: (x,y,w,h) or None
        """
        # Try YOLO TorchScript
        yolo_path = Path('models') / 'yolo.pt'
        if yolo_path.exists():
            try:
                import torch
                model = torch.jit.load(str(yolo_path))
                # run on a representative frame
                frames = sorted(self.frames_dir.glob('*.png'))
                if not frames:
                    return None
                img = cv2.imread(str(frames[len(frames)//2]))
                h, w = img.shape[:2]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # prepare tensor
                tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float()/255.0
                with torch.no_grad():
                    preds = model(tensor)
                # Expect preds as N x 6 [x1,y1,x2,y2,conf,class]
                try:
                    preds = preds[0].cpu().numpy()
                except Exception:
                    preds = preds.cpu().numpy()
                if preds.size == 0:
                    return None
                # pick highest-confidence detection
                best = max(preds, key=lambda r: r[4])
                x1,y1,x2,y2 = map(int, best[:4])
                return (x1, y1, x2-x1, y2-y1)
            except Exception:
                pass

        # Try EAST text detector
        east_path = Path('models') / 'frozen_east_text_detection.pb'
        if east_path.exists():
            try:
                # use OpenCV DNN EAST
                net = cv2.dnn.readNet(str(east_path))
                frames = sorted(self.frames_dir.glob('*.png'))
                if not frames:
                    return None
                img = cv2.imread(str(frames[len(frames)//2]))
                (H, W) = img.shape[:2]
                newW, newH = (320, 320)
                rW = W / float(newW)
                rH = H / float(newH)
                blob = cv2.dnn.blobFromImage(img, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
                net.setInput(blob)
                (scores, geometry) = net.forward(['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3'])
                # decode predictions
                (rects, confidences) = ([], [])
                (numRows, numCols) = scores.shape[2:4]
                for y in range(0, numRows):
                    scoresData = scores[0,0,y]
                    xData0 = geometry[0,0,y]
                    xData1 = geometry[0,1,y]
                    xData2 = geometry[0,2,y]
                    xData3 = geometry[0,3,y]
                    anglesData = geometry[0,4,y]
                    for x in range(0, numCols):
                        if scoresData[x] < 0.5:
                            continue
                        offsetX = x * 4.0
                        offsetY = y * 4.0
                        angle = anglesData[x]
                        cos = np.cos(angle)
                        sin = np.sin(angle)
                        h = xData0[x] + xData2[x]
                        w = xData1[x] + xData3[x]
                        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                        startX = int(endX - w)
                        startY = int(endY - h)
                        rects.append((startX, startY, endX, endY))
                        confidences.append(float(scoresData[x]))
                if rects:
                    # merge boxes and scale back
                    xs = [r[0] for r in rects]
                    ys = [r[1] for r in rects]
                    x2s = [r[2] for r in rects]
                    y2s = [r[3] for r in rects]
                    bx = int(min(xs) * rW)
                    by = int(min(ys) * rH)
                    bx2 = int(max(x2s) * rW)
                    by2 = int(max(y2s) * rH)
                    return (max(0,bx), max(0,by), min(W, bx2-bx), min(H, by2-by))
            except Exception:
                pass

        # Fallback heuristic (corner/edge-based)
        # sample first frame
        files = sorted(self.frames_dir.glob('*.png'))
        if not files:
            return None
        frame = cv2.imread(str(files[0]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        mask = np.zeros((h,w), dtype=np.uint8)
        regions = []
        for c in contours:
            x,y,cw,ch = cv2.boundingRect(c)
            area = cw*ch
            if area < 200:
                continue
            if (x < w*0.2 and y < h*0.2) or (x+cw > w*0.8 and y+ch > h*0.8) or (y+ch > h*0.75):
                regions.append((x,y,cw,ch))
        if regions:
            # return bounding union
            xs = [r[0] for r in regions]
            ys = [r[1] for r in regions]
            x2s = [r[0]+r[2] for r in regions]
            y2s = [r[1]+r[3] for r in regions]
            bx = min(xs)
            by = min(ys)
            bw = max(x2s)-bx
            bh = max(y2s)-by
            return (bx, by, bw, bh)

        return None

    def fuse_detections(self, boxes_list):
        """Fuse multiple lists of boxes (from different detectors) and return a combined bbox.

        boxes_list: iterable of lists of (x,y,w,h)
        """
        all_boxes = []
        for bl in boxes_list:
            if not bl:
                continue
            # if bl contains (x,y,w,h) tuples or list
            all_boxes.extend(list(bl))
        if not all_boxes:
            return None
        # Apply small dilation to each box and then NMS
        dilated = []
        for (x,y,w,h) in all_boxes:
            pad_w = int(w * 0.08) + 2
            pad_h = int(h * 0.08) + 2
            nx = max(0, x - pad_w)
            ny = max(0, y - pad_h)
            nw = w + pad_w*2
            nh = h + pad_h*2
            dilated.append((nx, ny, nw, nh))
        fused = _nms_boxes(dilated, iou_threshold=0.3)
        # return union of fused boxes
        xs = [b[0] for b in fused]
        ys = [b[1] for b in fused]
        x2s = [b[0]+b[2] for b in fused]
        y2s = [b[1]+b[3] for b in fused]
        bx = min(xs)
        by = min(ys)
        bw = max(x2s) - bx
        bh = max(y2s) - by
        return (bx, by, bw, bh)

    def preview_inpaint(self, frame_index: int = None, mask_coords: tuple = None, use_auto: bool = True, task: str = 'watermark') -> PreviewResult:
        """Generate a single-frame preview (inpainted) and save to work_dir/preview.png

        Returns PreviewResult with path to PNG
        """
        frames = sorted(self.frames_dir.glob('*.png'))
        if not frames:
            # extract frames to ensure at least one
            self.extract_frames()
            frames = sorted(self.frames_dir.glob('*.png'))
            if not frames:
                raise RuntimeError('No frames available for preview')

        if frame_index is None:
            idx = len(frames) // 2
        else:
            idx = min(max(0, frame_index), len(frames)-1)

        frame_path = frames[idx]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError('Failed to read frame')

        h, w = frame.shape[:2]

        # determine mask
        mask = None
        if mask_coords:
            x,y,ww,hh = mask_coords
            mm = np.zeros((h,w), dtype=np.uint8)
            mm[y:y+hh, x:x+ww] = 255
            mask = mm
        elif task == 'caption' and use_auto:
            boxes, combined = self.detect_captions()
            if combined:
                x,y,ww,hh = combined
                mm = np.zeros((h,w), dtype=np.uint8)
                mm[y:y+hh, x:x+ww] = 255
                mask = mm
        elif task == 'watermark' and use_auto:
            # try multiple detectors and fuse
            boxes_all = []
            try:
                wbox = self.detect_watermark()
                if wbox:
                    boxes_all.append([wbox])
            except Exception:
                pass
            # fallback: auto detect via edge heuristic (re-use auto_detect_mask style)
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                regs = []
                for cnt in contours:
                    x2,y2,cw,ch = cv2.boundingRect(cnt)
                    if cw*ch < 200:
                        continue
                    if (x2 < w*0.2 and y2 < h*0.2) or (x2+cw > w*0.8 and y2+ch > h*0.8) or (y2+ch > h*0.75):
                        regs.append((x2,y2,cw,ch))
                if regs:
                    boxes_all.append(regs)
            except Exception:
                pass

            fused = self.fuse_detections(boxes_all)
            if fused:
                x,y,ww,hh = fused
                mm = np.zeros((h,w), dtype=np.uint8)
                mm[y:y+hh, x:x+ww] = 255
                mask = mm

        if mask is None:
            # default empty mask -> nothing to preview
            mm = np.zeros((h,w), dtype=np.uint8)
            mask = mm

        # inpaint single frame
        out_img = _inpaint_with_optional_lama(frame, mask)
        preview_path = self.work_dir / 'preview.png'
        cv2.imwrite(str(preview_path), out_img)
        return PreviewResult(str(preview_path))

    def remove_soft_subtitles(self, output_no_subtitles: str):
        # Strip subtitle streams using -sn
        cmd = ['ffmpeg', '-y', '-i', str(self.input_path), '-c', 'copy', '-sn', str(output_no_subtitles)]
        subprocess.run(cmd, check=True)
        return output_no_subtitles

    def process_frames(self, mask_coords: tuple = None, use_auto: bool = True, progress_callback=None, max_workers: int = None):
        """Process frames in parallel using a ProcessPoolExecutor. progress_callback(processed_count, total_count)"""
        files = sorted(self.frames_dir.glob('*.png'))
        if not files:
            return

        total = len(files)
        max_workers = max_workers or min(32, (os.cpu_count() or 1))

        args_list = []
        for f in files:
            args_list.append((str(f), str(self.processed_dir), mask_coords, use_auto))

        processed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_process_single_frame, a): a for a in args_list}
            for fut in as_completed(futures):
                try:
                    frame_path, ok = fut.result()
                except Exception as e:
                    ok = False
                processed += 1
                if progress_callback:
                    try:
                        progress_callback(processed, total)
                    except Exception:
                        pass

    def reconstruct_video(self, remove_audio: bool = False, use_nvenc: bool = False):
        # Reconstruct using processed frames with multi-threaded FFmpeg and faster preset
        pattern = str(self.processed_dir / 'frame_%06d.png')
        temp_output = self.work_dir / 'temp_video.mp4'

        # Choose encoder
        if use_nvenc:
            # Try hardware encoder
            encoder = 'h264_nvenc'
        else:
            encoder = 'libx264'

        # Use faster preset, CRF for quality/speed tradeoff, and let FFmpeg choose threads
        cmd = ['ffmpeg', '-y', '-framerate', str(self.fps), '-i', pattern, '-c:v', encoder, '-preset', 'fast', '-crf', '23', '-threads', '0', '-pix_fmt', 'yuv420p', str(temp_output)]
        subprocess.run(cmd, check=True)

        # Add audio back if available and not removed
        final = self.output_path
        if not remove_audio and self.audio_path.exists():
            # Use copy of video stream if audio present
            cmd = ['ffmpeg', '-y', '-i', str(temp_output), '-i', str(self.audio_path), '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(final)]
            subprocess.run(cmd, check=True)
        else:
            shutil.move(str(temp_output), str(final))

        # Enforce resolution cap between 720p and 1080p (if greater than 1080p, scale down)
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(final)], capture_output=True, text=True)
        try:
            w_h = probe.stdout.strip().split('\n')[0].split(',')
            w, h = int(w_h[0]), int(w_h[1])
            if h > 1080:
                scaled = self.work_dir / 'scaled_output.mp4'
                cmd = ['ffmpeg', '-y', '-i', str(final), '-vf', 'scale=-2:1080', str(scaled)]
                subprocess.run(cmd, check=True)
                shutil.move(str(scaled), str(final))
            elif h < 720:
                # upscale to 720 preserving aspect
                scaled = self.work_dir / 'scaled_output.mp4'
                cmd = ['ffmpeg', '-y', '-i', str(final), '-vf', 'scale=-2:720', str(scaled)]
                subprocess.run(cmd, check=True)
                shutil.move(str(scaled), str(final))
        except Exception:
            pass

    def cleanup(self):
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def process(self, mask_coords: tuple = None, remove_audio: bool = False, use_auto: bool = True):
        try:
            self.extract_frames()
            self.extract_audio()
            self.process_frames(mask_coords=mask_coords, use_auto=use_auto)
            self.reconstruct_video(remove_audio=remove_audio)
        finally:
            # keep temp for debugging; caller may call cleanup()
            pass
