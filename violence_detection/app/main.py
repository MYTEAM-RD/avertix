from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Deque
from collections import deque
import numpy as np
import cv2
import sys
import torch
import os

# Set path dynamically to the project directory
path = "/home/ubuntu/violence_detection/violence_detection"
sys.path.append(path)

# Import utility functions and model loader
from app.utils import decode_frames, sample_decode_frames, decode_frame, sample_video_frames
from app.model_loader import load_model

# Initialize FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the violence detection model at startup
MODEL_PATH = "Nikeytas/videomae-crime-detector-ultra-v1"
try:
    model, processor = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# Determine the sequence length the model expects
SEQ_LENGTH = getattr(model.config, "num_frames", 16) if hasattr(model, "config") else 16

# Prepare a deque buffer for real-time frames
frame_buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LENGTH)

@app.get("/")
async def index():
    # Serve the realtime.html file from the static directory
    return FileResponse("static/realtime.html")

@app.post("/predict-realtime")
async def predict_realtime(frame: UploadFile = File(...)):
    try:
        image_bytes = await frame.read()
        img_bgr = decode_frame(image_bytes)
        if img_bgr is None:
            raise ValueError("Invalid image data")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame_buffer.append(img_rgb)

        violence_prob = 0.0
        if len(frame_buffer) == SEQ_LENGTH:
            frames_list = list(frame_buffer)
            inputs = processor(frames_list, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                violence_prob = probs[0][1].item()
        return {"violence_probability": violence_prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    try:
        contents = await video.read()
        if len(contents) == 0:
            raise ValueError("Empty video file")
        tmp_filename = f"temp_{video.filename}"
        with open(tmp_filename, "wb") as f:
            f.write(contents)
        frames = sample_video_frames(tmp_filename, num_frames=SEQ_LENGTH)
        if not frames:
            raise ValueError("Could not extract frames from video")
        inputs = processor(frames, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            violence_prob = probs[0][1].item()
        try:
            os.remove(tmp_filename)
        except OSError:
            pass
        return {"violence_probability": violence_prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")