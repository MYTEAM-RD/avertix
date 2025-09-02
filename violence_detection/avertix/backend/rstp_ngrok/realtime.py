import cv2
import time
import json
import threading
import subprocess
from flask import Flask, Response
from pathlib import Path
import logging
import requests

# --- CONFIG ---
with open("config.json") as f:
    config = json.load(f)

RTSP_URL_1 = config.get("camera1")
RTSP_URL_2 = config.get("camera2")
PORT = config.get("port", 8080)
HOST = "0.0.0.0"
STREAM_PATH_1 = "/stream1"
STREAM_PATH_2 = "/stream2"

# --- APP INIT ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FRAME BUFFERS ---
frame_buffers = {
    STREAM_PATH_1: {"frame": None, "lock": threading.Lock(), "running": True},
    STREAM_PATH_2: {"frame": None, "lock": threading.Lock(), "running": True}
}

# --- FRAME CAPTURE THREAD ---
def capture_frames(rtsp_url, stream_path):
    cap = None
    while frame_buffers[stream_path]["running"]:
        try:
            if cap is None or not cap.isOpened():
                logger.info(f"Connecting to {rtsp_url}")
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    logger.warning(f"Failed to open {rtsp_url}")
                    time.sleep(5)
                    continue

            success, frame = cap.read()
            if not success:
                logger.warning(f"Failed to read frame from {rtsp_url}")
                cap.release()
                cap = None
                time.sleep(5)
                continue

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                with frame_buffers[stream_path]["lock"]:
                    frame_buffers[stream_path]["frame"] = buffer.tobytes()

            time.sleep(0.033)
        except Exception as e:
            logger.error(f"Capture error on {rtsp_url}: {e}")
            if cap:
                cap.release()
                cap = None
            time.sleep(5)

    if cap:
        cap.release()

# --- GENERATE STREAM ---
def generate_frames(stream_path):
    while True:
        with frame_buffers[stream_path]["lock"]:
            frame = frame_buffers[stream_path]["frame"]

        if frame is None:
            yield (b'--frame\r\nContent-Type: text/plain\r\n\r\nWaiting for stream...\r\n')
            time.sleep(1)
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route(STREAM_PATH_1)
def stream1():
    return Response(generate_frames(STREAM_PATH_1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route(STREAM_PATH_2)
def stream2():
    return Response(generate_frames(STREAM_PATH_2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- START THREADS ---
def start_threads():
    threading.Thread(target=capture_frames, args=(RTSP_URL_1, STREAM_PATH_1), daemon=True).start()
    threading.Thread(target=capture_frames, args=(RTSP_URL_2, STREAM_PATH_2), daemon=True).start()

# --- START NGROK ---
def start_ngrok(port):
    ngrok_path = "/home/ubuntu/ngrok"
    subprocess.Popen([ngrok_path, "http", str(port)])
    time.sleep(3)
    try:
        res = requests.get("http://localhost:4040/api/tunnels").json()
        for t in res.get("tunnels", []):
            if t["proto"] == "https":
                public_url = t['public_url']
                logger.info(f"✅ Public URL: {public_url}/stream1 and /stream2")
                print(f"\n🌐 Access your streams at:\n  {public_url}/stream1\n  {public_url}/stream2\n")
    except Exception as e:
        logger.warning("Unable to fetch ngrok public URL")
        print("⚠️ Could not retrieve ngrok URL. Is ngrok running?")


# --- MAIN ---
if __name__ == "__main__":
    start_threads()
    start_ngrok(PORT)
    logger.info(f"Local: http://localhost:{PORT}/stream1 and /stream2")
    app.run(host=HOST, port=PORT, threaded=True)
