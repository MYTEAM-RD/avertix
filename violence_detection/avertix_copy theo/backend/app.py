# ==============================================================================
# FICHIER PRINCIPAL DE L'API DE DÉTECTION DE VIOLENCE
# ==============================================================================

# --- Imports Généraux et FastAPI ---
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import base64
import shutil
import tempfile
import cv2
import numpy as np
import os
import threading
import time
from typing import Dict
from collections import deque
from pydantic import BaseModel
import uuid

# --- Imports Spécifiques ---
import yt_dlp
from inference import ViolenceDetector # VOTRE VRAI MODÈLE EST IMPORTÉ ICI
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import httpx


# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialisation de FastAPI ---
app = FastAPI()

# --- MODIFICATION ---: Define path for temporary static videos and ensure it exists
STATIC_VIDEOS_DIR = "/home/ubuntu/violence_detection/violence_detection/avertixcopy/static_videos"
os.makedirs(STATIC_VIDEOS_DIR, exist_ok=True)

# --- MODIFICATION ---: Mount the new static directory to serve the preview videos
app.mount("/static_videos", StaticFiles(directory=STATIC_VIDEOS_DIR), name="static_videos")

app.mount("/assets", StaticFiles(directory="/home/ubuntu/violence_detection/violence_detection/avertixcopy/frontend/assets"), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODIFICATION ---: Background task for cleaning up old preview videos
CLEANUP_DELAY_SECONDS = 120 # 2 minutes

def cleanup_old_videos():
    """Periodically cleans up video files older than CLEANUP_DELAY_SECONDS."""
    while True:
        try:
            time.sleep(CLEANUP_DELAY_SECONDS / 2) # Check every 7.5 minutes
            now = time.time()
            logger.info("Running scheduled cleanup of old preview videos...")
            for filename in os.listdir(STATIC_VIDEOS_DIR):
                file_path = os.path.join(STATIC_VIDEOS_DIR, filename)
                if os.path.isfile(file_path):
                    # Check if file is older than the delay
                    if os.path.getmtime(file_path) < (now - CLEANUP_DELAY_SECONDS):
                        os.remove(file_path)
                        logger.info(f"Cleaned up old video file: {filename}")
        except Exception as e:
            logger.error(f"Error during video cleanup task: {e}")

@app.on_event("startup")
def startup_event():
    """Start the background cleanup thread when the app starts."""
    cleanup_thread = threading.Thread(target=cleanup_old_videos, daemon=True)
    cleanup_thread.start()
    logger.info("Background video cleanup task started.")
# --- END MODIFICATION ---

# ==============================================================================
# SECTION 1 : LOGIQUE DE STREAMING INTÉGRÉE
# ==============================================================================

# --- Classes Pydantic pour la configuration ---
class ModelConfig(BaseModel):
    clip_size: int
    memory: int
    threshold: int

class StartUpConfig(BaseModel):
    source: str
    modelConfig: ModelConfig

class YouTubeURL(BaseModel):
    url: str

# ==============================================================================
# --- NOUVELLE MODIFICATION : Pydantic model pour la requête de fichier local ---
class LocalFileRequest(BaseModel):
    path: str
# ==============================================================================

# --- Classe VideoCapture ---
class VideoCapture():
    def __init__(self, video_src):
        # La conversion en int pour les webcams est gérée ici
        if isinstance(video_src, str) and video_src.isdigit():
            source = int(video_src)
        else:
            source = video_src
            
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
             raise ValueError(f"Could not open video source: {source}")
        logger.info(f"Successfully opened video source: {source}")

        self.buffer = deque(maxlen=600) # Buffer pour les images brutes
        self.stop_flag = threading.Event()
        self.buffer_lock = threading.Lock()
        self.capture_thread = threading.Thread(target=self.capture)

    def isPlaying(self):
        return self.cap.isOpened()

    def capture(self):
        while self.isPlaying() and not self.stop_flag.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self.buffer_lock:
                    self.buffer.append(frame)
            else:
                # Si read() échoue, on arrête la capture
                self.cap.release()
                break
        self.cap.release()

    def read_clip(self, clip_size):
        clip = []
        # Attend que le buffer ait assez d'images, sauf si la capture est finie
        while len(self.buffer) < clip_size and self.isPlaying():
            time.sleep(0.01)

        with self.buffer_lock:
            while len(clip) < clip_size and len(self.buffer) > 0:
                clip.append(self.buffer.popleft())

        # Si le flux est terminé et qu'il manque des images, on boucle les dernières
        if not self.isPlaying() and 0 < len(clip) < clip_size:
            while len(clip) < clip_size:
                clip.extend(clip[: clip_size - len(clip) ])
        return clip

    def start_capture_thread(self):
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def end_capture_thread(self):
        self.stop_flag.set()
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

    def isFlowing(self):
        return self.isPlaying() or len(self.buffer) > 0

# --- Classe Controller ---
class Controller():
    def __init__(self):
        # Utilise votre vraie classe ViolenceDetector
        self.model = ViolenceDetector(model_dir="/home/ubuntu/violence_detection/violence_detection/model")
        self.video_capture = None
        self.processing_thread = None
        self.stop_flag = None

        # La fonctionnalité de 'OutputPipe' et 'PreformanceTimer' est maintenant gérée ici
        self.output_buffer = deque(maxlen=300) # Buffer pour les images traitées (pour le streaming)
        self.frame_rate = 0.0
        self.last_prediction = {}

    def start(self, config: StartUpConfig):
        self.end()
        source_url = config.source
        if 'youtube.com' in source_url or 'youtu.be' in source_url:
            logger.info("YouTube link detected. Attempting to extract direct stream URL with cookies...")
            try:
                COOKIES_FILE_PATH = "/home/ubuntu/violence_detection/violence_detection/avertixcopy/cookies.txt"

                if not os.path.exists(COOKIES_FILE_PATH):
                    logger.warning(f"Cookies file not found at {COOKIES_FILE_PATH}. Proceeding without authentication. This may fail.")
                    ydl_opts = {
                        'format': 'best[ext=mp4]/best',
                    }
                else:
                    ydl_opts = {
                        'format': 'best[ext=mp4]/best',
                        'cookiefile': COOKIES_FILE_PATH,
                    }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(source_url, download=False)
                    direct_url = info_dict.get('url')
                    if not direct_url:
                        raise ValueError("yt-dlp could not find a stream URL.")
                    logger.info("Successfully extracted direct URL.")
                    config.source = direct_url
            except Exception as e:
                logger.error(f"Error extracting YouTube URL: {e}")
                raise ValueError(f"Failed to process YouTube URL. Error: {e}")
        
        if hasattr(self.model, 'update_config'):
             self.model.update_config(config.modelConfig.dict())
        else:
             logger.warning("Model does not have 'update_config' method. Using existing settings.")

        self.last_prediction = {}
        self.output_buffer.clear()
        
        self.video_capture = VideoCapture(video_src=config.source)
        self.video_capture.start_capture_thread()
        
        self.stop_flag = threading.Event()
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def processing_loop(self):
        start_time = time.time()
        processed_frames_count = 0
        clip_size = 16 

        while self.video_capture.isFlowing() and not self.stop_flag.is_set():
            clip = self.video_capture.read_clip(clip_size)
            if not clip:
                break
            
            prediction_result = self.model.predict_clip(clip)
            self.last_prediction = {
                "label": "Violence" if prediction_result > 0.5 else "Non-Violence",
                "confidence": float(prediction_result)
            }
            
            for frame in clip:
                text = f"{self.last_prediction['label']} ({self.last_prediction['confidence']:.2f})"
                color = (0, 0, 255) if self.last_prediction['label'] == 'Violence' else (0, 255, 0)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                self.output_buffer.append(frame)

            processed_frames_count += len(clip)
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                self.frame_rate = processed_frames_count / elapsed_time
        
        if self.video_capture:
            self.video_capture.end_capture_thread()

    def end(self):
        if self.stop_flag:
            self.stop_flag.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        if self.video_capture:
            self.video_capture.end_capture_thread()
            
        self.output_buffer.clear()
        logger.info("Stream ended successfully.")

    def getModelConfig(self):
        return {
            "clip_size": getattr(self.model, 'clip_size', 16),
            "threshold": getattr(self.model, 'threshold', 0.5)
        }

    def stream(self):
        """Générateur qui produit le flux MJPEG."""
        while self.stop_flag is None or not self.stop_flag.is_set():
            if len(self.output_buffer) > 0:
                frame = self.output_buffer.popleft()
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')
            else:
                time.sleep(0.01)

# --- Instance globale du Controller ---
controller = Controller()

# ==============================================================================
# SECTION 2 : NOUVEAUX ENDPOINTS
# ==============================================================================

# --- Buffers temps réel ---
single_buffer: list[np.ndarray] = []
multi_stream_buffer: Dict[int, list[np.ndarray]] = {}

# --- Instance de détecteur pour l'analyse de fichiers statiques ---
static_detector = ViolenceDetector(model_dir="/home/ubuntu/violence_detection/violence_detection/model")

# --- Fonction d'analyse de fichier statique ---
def analyze_video_file(file_path: str) -> dict:
    try:
        vid = cv2.VideoCapture(file_path)
        if not vid.isOpened(): raise ValueError("Could not open video file")
        
        fps = vid.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frames, predictions = [], []
        
        while True:
            success, frame = vid.read()
            if not success: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

            if len(frames) == 16:
                pred = static_detector.predict_clip(frames)
                predictions.append(float(pred))
                frames = []
        vid.release()
        return {"predictions": predictions, "fps": fps, "total_frames": total_frames}
        
    except Exception as e:
        logger.error(f"Error analyzing video file: {e}")
        raise

# --- Endpoints ---

@app.get("/")
def serve_index():
    return FileResponse("/home/ubuntu/violence_detection/violence_detection/avertixcopy/frontend/MergedAlert.html")

# --- Nouveaux Endpoints pour le Streaming Controller ---
@app.post("/stream/start")
async def start_stream_processing(config: StartUpConfig):
    try:
        controller.start(config)
        return JSONResponse({"status": "success", "message": "Stream processing started."})
    except Exception as e:
        logger.error(f"Failed to start stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream/stop")
async def stop_stream_processing():
    controller.end()
    return JSONResponse({"status": "success", "message": "Stream processing stopped."})

@app.get("/stream/feed")
def get_video_feed():
    return StreamingResponse(controller.stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/status")
async def get_stream_status():
    return JSONResponse({
        "is_active": controller.processing_thread.is_alive() if controller.processing_thread else False,
        "framerate": round(controller.frame_rate, 2),
        "last_prediction": controller.last_prediction
    })

@app.get("/stream/config")
async def get_stream_config():
    return JSONResponse(controller.getModelConfig())

@app.post("/analyze-upload/")
async def analyze_uploaded(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        temp_file_path = tmp.name
        shutil.copyfileobj(file.file, tmp)
    try:
        result = analyze_video_file(temp_file_path)
        return JSONResponse(content=result)
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# ==============================================================================
# ENDPOINT POUR L'ANALYSE OFFLINE DE VIDÉOS YOUTUBE 
# ==============================================================================
@app.post("/analyze-youtube-offline/")
async def analyze_youtube_offline(data: YouTubeURL):
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name

        COOKIES_FILE_PATH = "/home/ubuntu/violence_detection/violence_detection/avertixcopy/cookies.txt"
        ydl_opts = {
            'format': 'best[ext=mp4][height<=720]/best[height<=720]',
            'outtmpl': temp_file_path,
            'overwrites': True,
            'quiet': True,
        }
        
        if os.path.exists(COOKIES_FILE_PATH):
            logger.info(f"Using cookies file for offline download: {COOKIES_FILE_PATH}")
            ydl_opts['cookiefile'] = COOKIES_FILE_PATH
        else:
            logger.warning(f"Cookies file not found at {COOKIES_FILE_PATH}. Download may fail for private/age-restricted videos.")

        logger.info(f"Downloading YouTube video from {data.url} to {temp_file_path}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([data.url])
        logger.info("YouTube video download complete.")

        logger.info(f"Analyzing downloaded file: {temp_file_path}")
        result = analyze_video_file(temp_file_path)
        logger.info("Analysis of downloaded file complete.")

        unique_filename = f"{uuid.uuid4()}.mp4"
        destination_path = os.path.join(STATIC_VIDEOS_DIR, unique_filename)
        shutil.move(temp_file_path, destination_path)
        logger.info(f"Moved video to static directory for preview: {destination_path}")

        result['video_url'] = f"/static_videos/{unique_filename}"
        
        return JSONResponse(content=result)

    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        logger.error(f"Error during YouTube offline analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process YouTube video: {str(e)}")
# ==============================================================================


# ==============================================================================
# --- NOUVELLE MODIFICATION : ENDPOINT POUR L'ANALYSE DE FICHIER LOCAL ---
# ==============================================================================
@app.post("/analyze-local-file/")
async def analyze_local_file(data: LocalFileRequest):
    """
    Analyzes a video file that is already present on the server's local filesystem.
    This is used for the pre-selected test videos in the frontend sidebar.
    NOTE: Be cautious with this endpoint in production, as it accesses the local
    filesystem based on a path provided by the client. Ensure proper validation
    if exposed to untrusted users.
    """
    source_path = data.path
    
    # Basic security check to ensure the file exists and is a file
    if not os.path.exists(source_path) or not os.path.isfile(source_path):
        logger.error(f"Local file not found or is not a file: {source_path}")
        raise HTTPException(status_code=404, detail=f"Local file not found at path: {source_path}")

    try:
        # Analyze the local file using the existing helper function
        logger.info(f"Analyzing local server file: {source_path}")
        result = analyze_video_file(source_path)
        logger.info("Analysis of local file complete.")

        # Copy the video to the static directory so it can be previewed in the browser.
        # We use copy() instead of move() to preserve the original test file.
        unique_filename = f"{uuid.uuid4()}.mp4"
        destination_path = os.path.join(STATIC_VIDEOS_DIR, unique_filename)
        shutil.copy(source_path, destination_path)
        logger.info(f"Copied local video to static directory for preview: {destination_path}")

        # Add the web-accessible URL to the response, which the frontend needs
        result['video_url'] = f"/static_videos/{unique_filename}"
        
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error during local file analysis for path {source_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process local file: {str(e)}")
# ==============================================================================


@app.post("/realtime-frame/")
async def realtime_frame(request: Request):
    global single_buffer
    try:
        frame_b64 = (await request.json()).get("frame")
        img = cv2.imdecode(np.frombuffer(base64.b64decode(frame_b64), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        single_buffer.append(img)
        if len(single_buffer) > 16: single_buffer.pop(0)

        prediction = float(static_detector.predict_clip(single_buffer)) if len(single_buffer) == 16 else 0.0
        return JSONResponse({"success": True, "prediction": prediction})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to process frame: {str(e)}"})

@app.post("/realtime-cam/")
async def realtime_cam(request: Request):
    global multi_stream_buffer
    try:
        data = await request.json()
        frame_b64, stream_id = data.get("frame"), data.get("stream_id", 1)
        if stream_id not in multi_stream_buffer: multi_stream_buffer[stream_id] = []
        
        img = cv2.imdecode(np.frombuffer(base64.b64decode(frame_b64), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        multi_stream_buffer[stream_id].append(img)
        if len(multi_stream_buffer[stream_id]) > 16: multi_stream_buffer[stream_id].pop(0)
        
        prediction = float(static_detector.predict_clip(multi_stream_buffer[stream_id])) if len(multi_stream_buffer[stream_id]) == 16 else 0.0
        return JSONResponse({"success": True, "prediction": prediction})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to process frame: {str(e)}"})

@app.get("/proxy/")
async def proxy(url: str):
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=30)
        return StreamingResponse(resp.aiter_bytes(), media_type=resp.headers.get('content-type'))

@app.post("/kinesis-stream/")
async def get_kinesis_stream_url(request: Request):
    try:
        arn = (await request.json()).get("arn")
        client = boto3.client("kinesisvideo")
        endpoint = client.get_data_endpoint(StreamARN=arn, APIName="GET_HLS_STREAMING_SESSION_URL")["DataEndpoint"]
        kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint)
        hls_stream = kvam.get_hls_streaming_session_url(StreamARN=arn, PlaybackMode="LIVE", Expires=300)
        return JSONResponse({"success": True, "streamUrl": hls_stream["HLSStreamingSessionURL"]})
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"AWS Error: {str(e)}")

@app.get("/health/")
async def health_check():
    return JSONResponse({
        "success": True, "status": "healthy",
        "streaming_active": controller.processing_thread.is_alive() if controller.processing_thread else False
    })

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API... stopping any active stream.")
    controller.end()