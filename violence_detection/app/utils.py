import cv2
import numpy as np

def decode_frame(image_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into a NumPy BGR image using OpenCV.
    Returns None if decoding fails.
    """
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return frame  # BGR format for OpenCV models

def sample_video_frames(video_path: str, num_frames: int = 16) -> list:
    """
    Extracts a list of `num_frames` frames from the video at `video_path`, evenly spaced from start to end.
    Returns a list of frames in RGB format.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []  # could not open video file

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        step = 1
        idx = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            idx += 1
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames

# >>>> 🔧 ADD THE MISSING FUNCTIONS BELOW <<<

def decode_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def sample_decode_frames(video_path, sample_rate=5):
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames
