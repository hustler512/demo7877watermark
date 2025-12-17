const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const previewVideo = document.getElementById('previewVideo');
const overlayCanvas = document.getElementById('overlayCanvas');
const autoBtn = document.getElementById('autoBtn');
const startBtn = document.getElementById('startBtn');
const taskSelect = document.getElementById('task');
const autoDetect = document.getElementById('autoDetect');
const removeAudio = document.getElementById('removeAudio');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const downloadLink = document.getElementById('downloadLink');
const previewBtn = document.getElementById('previewBtn');
const previewImg = document.getElementById('previewImg');

let currentJobId = null;
let uploadedFile = null;
let drawState = {dragging:false, startX:0, startY:0, rect:null};

function resizeCanvas() {
  const rect = previewVideo.getBoundingClientRect();
  overlayCanvas.width = rect.width;
  overlayCanvas.height = rect.height;
  overlayCanvas.style.left = rect.left + 'px';
  overlayCanvas.style.top = rect.top + 'px';
}

function clearOverlay(){
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0,0,overlayCanvas.width, overlayCanvas.height);
}

function drawRectOnOverlay(x,y,w,h){
  const ctx = overlayCanvas.getContext('2d');
  clearOverlay();
  ctx.strokeStyle = '#00a0ff';
  ctx.lineWidth = 2;
  ctx.setLineDash([6,4]);
  ctx.strokeRect(x,y,w,h);
  ctx.fillStyle = 'rgba(0,160,255,0.12)';
  ctx.fillRect(x,y,w,h);
}

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    uploadedFile = e.target.files[0];
    fileInfo.textContent = uploadedFile.name + ' — ' + (uploadedFile.size / (1024*1024)).toFixed(2) + ' MB';
    // show preview
    const url = URL.createObjectURL(uploadedFile);
    previewVideo.src = url;
    previewVideo.addEventListener('loadedmetadata', () => {
      // set overlay size
      resizeCanvas();
    });
  }
});

uploadBtn.addEventListener('click', async () => {
  if (!uploadedFile) return alert('Select a file first');
  const fd = new FormData();
  fd.append('file', uploadedFile);
  uploadBtn.disabled = true;
  uploadBtn.textContent = 'Uploading...';
  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');
    currentJobId = data.job_id;
    alert('Upload successful. Job ID: ' + currentJobId);
  } catch (err) {
    alert(err.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = 'Upload';
  }
});

window.addEventListener('resize', () => { try{ resizeCanvas(); }catch(e){} });

// Canvas drawing events for manual mask
overlayCanvas.addEventListener('mousedown', (e) => {
  if (autoDetect.checked) return;
  drawState.dragging = true;
  const rect = overlayCanvas.getBoundingClientRect();
  drawState.startX = e.clientX - rect.left;
  drawState.startY = e.clientY - rect.top;
});

overlayCanvas.addEventListener('mousemove', (e) => {
  if (!drawState.dragging) return;
  const rect = overlayCanvas.getBoundingClientRect();
  const x = drawState.startX;
  const y = drawState.startY;
  const w = (e.clientX - rect.left) - x;
  const h = (e.clientY - rect.top) - y;
  drawState.rect = {x: Math.min(x, x+w), y: Math.min(y, y+h), w: Math.abs(w), h: Math.abs(h)};
  drawRectOnOverlay(drawState.rect.x, drawState.rect.y, drawState.rect.w, drawState.rect.h);
});

overlayCanvas.addEventListener('mouseup', (e) => {
  if (!drawState.dragging) return;
  drawState.dragging = false;
  if (!drawState.rect) return;
  // populate manual fields (scaled to video natural size)
  const rect = previewVideo.getBoundingClientRect();
  const scaleX = previewVideo.videoWidth / rect.width;
  const scaleY = previewVideo.videoHeight / rect.height;
  const mx = Math.round(drawState.rect.x * scaleX);
  const my = Math.round(drawState.rect.y * scaleY);
  const mw = Math.round(drawState.rect.w * scaleX);
  const mh = Math.round(drawState.rect.h * scaleY);
  document.getElementById('mask_x').value = mx;
  document.getElementById('mask_y').value = my;
  document.getElementById('mask_w').value = mw;
  document.getElementById('mask_h').value = mh;
});

// Auto-detect button: trigger process with auto_detect=true and ask backend to detect mask
autoBtn.addEventListener('click', async () => {
  if (!currentJobId) return alert('Upload a video first');
  const fd = new FormData();
  fd.append('job_id', currentJobId);
  fd.append('task', taskSelect.value);
  fd.append('auto_detect', true);
  fd.append('remove_audio', removeAudio.checked);
  // send request to process which will auto-detect; we just start and poll
  startBtn.disabled = true;
  startBtn.textContent = 'Processing...';
  const res = await fetch('/process', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) { alert(data.error || 'Processing failed'); startBtn.disabled=false; startBtn.textContent='Start Processing'; return; }
  const interval = setInterval(async () => {
    const s = await fetch('/status/' + currentJobId);
    const j = await s.json();
    progressFill.style.width = (j.progress || 0) + '%';
    progressText.textContent = (j.progress || 0) + '% — ' + (j.status || '');
    if (j.status === 'completed') {
      clearInterval(interval);
      downloadLink.style.display = 'inline-block';
      downloadLink.href = '/download/' + currentJobId;
      downloadLink.textContent = 'Download Clean Video';
      startBtn.disabled = false;
      startBtn.textContent = 'Start Processing';
    }
  }, 1500);
});

// Preview button: request backend preview and display PNG
previewBtn.addEventListener('click', async () => {
  if (!currentJobId) return alert('Upload a video first');
  previewBtn.disabled = true;
  const oldText = progressText.textContent;
  progressText.textContent = 'Generating preview...';
  previewImg.style.display = 'none';

  try {
    const mx = document.getElementById('mask_x').value;
    const my = document.getElementById('mask_y').value;
    const mw = document.getElementById('mask_w').value;
    const mh = document.getElementById('mask_h').value;
    const params = new URLSearchParams();
    params.set('task', taskSelect.value);
    params.set('use_auto', autoDetect.checked ? 'true' : 'false');
    if (mx && my && mw && mh) {
      params.set('mask_x', mx);
      params.set('mask_y', my);
      params.set('mask_w', mw);
      params.set('mask_h', mh);
    }
    const url = '/preview/' + encodeURIComponent(currentJobId) + '?' + params.toString();
    const res = await fetch(url);
    if (!res.ok) {
      let err = 'Preview failed';
      try { const j = await res.json(); err = j.error || err; } catch(e){}
      throw new Error(err);
    }
    const blob = await res.blob();
    const imgUrl = URL.createObjectURL(blob);
    previewImg.src = imgUrl;
    previewImg.style.display = 'block';
    // clear overlay while showing preview
    clearOverlay();
  } catch (err) {
    alert(err.message || 'Preview error');
  } finally {
    previewBtn.disabled = false;
    progressText.textContent = oldText;
  }
});

startBtn.addEventListener('click', async () => {
  if (!currentJobId) return alert('Upload a video first');
  const fd = new FormData();
  fd.append('job_id', currentJobId);
  fd.append('task', taskSelect.value);
  fd.append('auto_detect', autoDetect.checked);
  fd.append('remove_audio', removeAudio.checked);
  // Manual mask fields
  const mx = document.getElementById('mask_x').value;
  const my = document.getElementById('mask_y').value;
  const mw = document.getElementById('mask_w').value;
  const mh = document.getElementById('mask_h').value;
  if (!autoDetect.checked && mx && my && mw && mh) {
    fd.append('mask_x', mx);
    fd.append('mask_y', my);
    fd.append('mask_w', mw);
    fd.append('mask_h', mh);
  }

  startBtn.disabled = true;
  startBtn.textContent = 'Processing...';

  const res = await fetch('/process', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || 'Processing failed');
    startBtn.disabled = false;
    startBtn.textContent = 'Start Processing';
    return;
  }

  if (data.status === 'enqueued') {
    alert('Job enqueued to Redis worker. Job ID: ' + data.job_id);
  }

  // Poll status
  const interval = setInterval(async () => {
    const s = await fetch('/status/' + currentJobId);
    const j = await s.json();
    progressFill.style.width = (j.progress || 0) + '%';
    progressText.textContent = (j.progress || 0) + '% — ' + (j.status || '');
    if (j.status === 'completed') {
      clearInterval(interval);
      downloadLink.style.display = 'inline-block';
      downloadLink.href = '/download/' + currentJobId;
      downloadLink.textContent = 'Download Clean Video';
      startBtn.disabled = false;
      startBtn.textContent = 'Start Processing';
    } else if (j.status === 'failed') {
      clearInterval(interval);
      alert('Processing failed: ' + (j.error || 'unknown error'));
      startBtn.disabled = false;
      startBtn.textContent = 'Start Processing';
    }
  }, 1500);
});
