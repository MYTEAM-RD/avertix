# backend/app.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import base64
import shutil
import tempfile
import requests
import cv2
import numpy as np
from typing import Dict
from inference import ViolenceDetector

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount static frontend assets
app.mount("/assets", StaticFiles(directory="/home/ubuntu/violence_detection/violence_detection/avertix/frontend/assets"), name="assets")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the violence detection model
detector = ViolenceDetector(model_dir="/home/ubuntu/violence_detection/violence_detection/model")

# Buffers
single_buffer: list[np.ndarray] = []  # For /realtime-frame (single stream)
multi_stream_buffer: Dict[int, list[np.ndarray]] = {}  # For /realtime-cam (multi camera)

# ------------------------------------------------------------
# 1. Serve frontend
# ------------------------------------------------------------
@app.get("/")
def serve_index():
    return FileResponse("/home/ubuntu/violence_detection/violence_detection/avertix/frontend/index.html")

# ------------------------------------------------------------
# 2. Analyze uploaded video
# ------------------------------------------------------------
@app.post("/analyze-upload/")
async def analyze_uploaded(file: UploadFile = File(...)):
    """Analyse an uploaded video and return violence probability per clip."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)

    vid = cv2.VideoCapture(tmp.name)
    fps = vid.get(cv2.CAP_PROP_FPS) or 30
    frames, predictions = [], []

    while True:
        success, frame = vid.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

        if len(frames) == 16:
            pred = detector.predict_clip(frames)
            predictions.append(float(pred))
            frames = []

    vid.release()
    return JSONResponse(content={"predictions": predictions, "fps": fps})

# ------------------------------------------------------------
# 3. Realtime (single frame analysis)
# ------------------------------------------------------------
@app.post("/realtime-frame/")
async def realtime_frame(request: Request):
    """Receive one base64-JPEG frame, return instantaneous probability."""
    global single_buffer
    frame_b64 = (await request.json()).get("frame")

    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(frame_b64), np.uint8),
        cv2.IMREAD_COLOR,
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    single_buffer.append(img)
    if len(single_buffer) > 16:
        single_buffer.pop(0)

    prediction = float(detector.predict_clip(single_buffer)) if len(single_buffer) == 16 else 0.0
    return JSONResponse({"prediction": prediction})

# ------------------------------------------------------------
# 4. Realtime from IP camera (multi-stream)
# ------------------------------------------------------------
@app.post("/realtime-cam")
async def realtime_cam(request: Request):
    """Receive one base64-JPEG frame from IP camera stream, return instantaneous probability."""
    global multi_stream_buffer
    try:
        data = await request.json()
        frame_b64 = data.get("frame")
        stream_id = data.get("stream_id", 1)

        if not frame_b64:
            raise ValueError("No frame provided in request")

        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(frame_b64), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if img is None:
            raise ValueError("Failed to decode image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        if stream_id not in multi_stream_buffer:
            multi_stream_buffer[stream_id] = []

        multi_stream_buffer[stream_id].append(img)
        if len(multi_stream_buffer[stream_id]) > 16:
            multi_stream_buffer[stream_id].pop(0)

        prediction = float(detector.predict_clip(multi_stream_buffer[stream_id])) if len(multi_stream_buffer[stream_id]) == 16 else 0.0
        logger.info(f"Prediction for stream_id {stream_id}: {prediction}")
        return JSONResponse({"prediction": prediction})

    except Exception as e:
        logger.error(f"Error processing frame for stream_id {stream_id}: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Failed to process frame: {str(e)}"})

# ------------------------------------------------------------
# 5. Proxy to stream MJPEG from IP cameras
# ------------------------------------------------------------
@app.get("/proxy/")
async def proxy(url: str):
    try:
        headers = {'ngrok-skip-browser-warning': 'true'}
        resp = requests.get(url, headers=headers, stream=True)
        if not resp.ok:
            return Response(f"Error: HTTP {resp.status_code}", status_code=resp.status_code)
        return StreamingResponse(resp.iter_content(chunk_size=10 * 1024),
                                 media_type=resp.headers.get('content-type', 'application/octet-stream'))
    except Exception as e:
        return Response(f"Proxy error: {str(e)}", status_code=500)
