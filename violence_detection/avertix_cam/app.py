import os
import cv2
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import time
import torch
import gradio as gr
import soundfile as sf
from transformers import VideoMAEForVideoClassification, AutoImageProcessor
import threading
import queue

# ── Model Initialization ─────────────────────────────────────────────────
MODEL_DIR = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(MODEL_DIR, use_fast=False)
model.eval()

# Thread lock for model access to prevent concurrent CUDA issues
model_lock = threading.Lock()

# ── Constants ───────────────────────────────────────────────────────────
CLIP = 16
STRIDE = 1
STREAM_FPS = 15
HIST_LEN = 60

# ── Sliding window for video file ───────────────────────────────────────
def sliding_window(path, clip_len=CLIP, stride=1):
    cap, frames = cv2.VideoCapture(path), []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) / 255.0)
    cap.release()

    probs = []
    for s in range(0, len(frames) - clip_len + 1, stride):
        clip = [cv2.resize(frames[i], (224, 224)) for i in range(s, s + clip_len)]
        inp = processor(clip, return_tensors="pt", do_rescale=False).to(DEVICE)
        with torch.no_grad():
            with model_lock:
                p = torch.softmax(model(**inp).logits, dim=1)[0, 1].item()
        probs.append(p)
    return frames, probs, fps

def make_outputs(frames, probs, fps, sensitivity):
    frames_uint8 = [(f * 255.0).astype(np.uint8) for f in frames]
    h, w = frames_uint8[0].shape[:2]
    for idx, p in enumerate(probs):
        if p > sensitivity:
            for f in range(idx, min(idx + CLIP, len(frames_uint8))):
                cv2.rectangle(frames_uint8[f], (0, 0), (w - 1, h - 1), (255, 0, 0), 6)

    out_vid = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    vw = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames_uint8:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    t = np.arange(len(probs)) / fps
    fig, ax = plt.subplots(figsize=(5, 2))
    colors = ['red' if p > sensitivity else 'blue' for p in probs]
    ax.bar(t, probs, color=colors, width=1/fps)
    ax.set_ylim(0, 1)
    ax.set_xlabel('sec'); ax.set_ylabel('fight prob')
    fig.tight_layout()
    out_plot = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    fig.savefig(out_plot, dpi=120); plt.close(fig)

    out_beep = None
    if any(p > sensitivity for p in probs):
        sr = 16000
        tone = 0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))
        out_beep = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        sf.write(out_beep, tone, sr)

    return out_vid, out_plot, out_beep

def run_pipeline(video_path, sensitivity):
    if not video_path or not os.path.exists(video_path):
        return None, None, None
    frames, probs, fps = sliding_window(video_path)
    return make_outputs(frames, probs, fps, sensitivity)

# ── Utilities ───────────────────────────────────────────────────────────
def resize_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    return frame / 255.0

def plot_history(history, sensitivity):
    fig, ax = plt.subplots(figsize=(5, 2))
    colors = ['red' if p > sensitivity else 'blue' for p in history]
    ax.bar(np.arange(len(history)), history, color=colors)
    ax.set_ylim(0, 1)
    ax.set_xlabel('recent clips'); ax.set_ylabel('fight prob')
    fig.tight_layout()
    return fig

# ── Live webcam handler ─────────────────────────────────────────────────
last_process_time = 0
def live_handler(frame, state, history, sensitivity, stream_fps):
    global last_process_time
    current_time = time.time()
    if current_time - last_process_time < 1 / stream_fps:
        return state, frame, history, plot_history(history, sensitivity), 0.0, "Throttled"
    last_process_time = current_time
    if frame is None:
        return state, frame, history, plot_history(history, sensitivity), 0.0, "No frame"
    frame = resize_frame(frame)
    state.append(frame)
    if len(state) < CLIP:
        return state, frame, history, plot_history(history, sensitivity), 0.0, "Buffering"

    result_queue = queue.Queue()
    def process_clip():
        clip = list(state)[-CLIP:]
        inp = processor(clip, return_tensors="pt", do_rescale=False).to(DEVICE)
        with torch.no_grad():
            with model_lock:
                p = torch.softmax(model(**inp).logits, dim=1)[0, 1].item()
        del inp
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        result_queue.put(p)

    thread = threading.Thread(target=process_clip)
    thread.start()
    thread.join()
    p = result_queue.get()

    history.append(p)
    if len(history) > HIST_LEN:
        history.pop(0)
    smoothed_p = np.mean(history[-5:]) if len(history) >= 5 else p
    disp = (frame * 255.0).astype(np.uint8)
    h, w = disp.shape[:2]
    if smoothed_p > sensitivity:
        cv2.rectangle(disp, (0, 0), (w-1, h-1), (255, 0, 0), 6)
    cv2.putText(disp, f'fight prob: {smoothed_p:.2f}', (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return state, disp, history, plot_history(history, sensitivity), smoothed_p, "Processed"

# ── HTTP Stream Handler 1 ───────────────────────────────────────────────
last_http_process_time = 0
def http_stream_handler(http_url, state, history, sensitivity, stream_fps):
    """Process HTTP stream frames with model inference in a thread."""
    global last_http_process_time
    current_time = time.time()
    if current_time - last_http_process_time < 1/stream_fps:
        return state, None, history, plot_history(history, sensitivity), history[-1] if history else 0.0, "Throttled"
    
    last_http_process_time = current_time
    if not http_url:
        return state, None, history, plot_history(history, sensitivity), history[-1] if history else 0.0, "No HTTP URL"
    
    cap = cv2.VideoCapture(http_url)
    if not cap.isOpened():
        return state, None, history, plot_history(history, sensitivity), history[-1] if history else 0.0, "Failed to open HTTP stream"
    
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return state, None, history, plot_history(history, sensitivity), history[-1] if history else 0.0, "No frame from HTTP stream"
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = resize_frame(frame)
    state.append(frame)
    
    if len(state) < CLIP:
        return state, frame, history, plot_history(history, sensitivity), 0.0, "Waiting for enough frames"
    
    result_queue = queue.Queue()
    def process_clip():
        clip_buffer = list(state)[-CLIP:]
        inp = processor(clip_buffer, return_tensors='pt', do_rescale=False).to(DEVICE)
        with torch.no_grad():
            with model_lock:
                p = torch.softmax(model(**inp).logits, dim=1)[0,1].item()
        del inp
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        result_queue.put(p)
    
    thread = threading.Thread(target=process_clip)
    thread.start()
    thread.join()
    p = result_queue.get()
    
    history.append(p)
    if len(history) > HIST_LEN:
        history.pop(0)
    
    smoothed_p = np.mean(history[-5:]) if len(history) >= 5 else p
    disp = (frame * 255.0).astype(np.uint8)
    h, w = disp.shape[:2]
    if smoothed_p > sensitivity:
        cv2.rectangle(disp, (0,0), (w-1,h-1), (255,0,0), 6)
    cv2.putText(disp, f'fight prob: {smoothed_p:.2f}', (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    
    return state, disp, history, plot_history(history, sensitivity), smoothed_p, "Processed"

# ── HTTP Stream Handler 2 ───────────────────────────────────────────────
last_http_process_time2 = 0
def http_stream_handler2(http_url, state, history, sensitivity, stream_fps):
    """Process second HTTP stream frames with model inference in a thread."""
    global last_http_process_time2
    current_time = time.time()
    if current_time - last_http_process_time2 < 1/stream_fps:
        return state, None, history, plot_history(history, sensitivity), history[-1] if history else 0.0, "Throttled"
    
    last_http_process_time2 = current_time
    if not http_url:
        return state, None, history, plot_history(history, sensitivity), history[-1] if history else 0.0, "No HTTP URL"
    
    cap = cv2.VideoCapture(http_url)
    if not cap.isOpened():
        return state, None, history, plot_history(history, sensitivity), history[-1] if history else 0.0, "Failed to open HTTP stream"
    
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return state, None, history, plot_history(history, sensitivity), history[-1] if history else 0.0, "No frame from HTTP stream"
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = resize_frame(frame)
    state.append(frame)
    
    if len(state) < CLIP:
        return state, frame, history, plot_history(history, sensitivity), 0.0, "Waiting for enough frames"
    
    result_queue = queue.Queue()
    def process_clip():
        clip_buffer = list(state)[-CLIP:]
        inp = processor(clip_buffer, return_tensors='pt', do_rescale=False).to(DEVICE)
        with torch.no_grad():
            with model_lock:
                p = torch.softmax(model(**inp).logits, dim=1)[0,1].item()
        del inp
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        result_queue.put(p)
    
    thread = threading.Thread(target=process_clip)
    thread.start()
    thread.join()
    p = result_queue.get()
    
    history.append(p)
    if len(history) > HIST_LEN:
        history.pop(0)
    
    smoothed_p = np.mean(history[-5:]) if len(history) >= 5 else p
    disp = (frame * 255.0).astype(np.uint8)
    h, w = disp.shape[:2]
    if smoothed_p > sensitivity:
        cv2.rectangle(disp, (0,0), (w-1,h-1), (255,0,0), 6)
    cv2.putText(disp, f'fight prob: {smoothed_p:.2f}', (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    
    return state, disp, history, plot_history(history, sensitivity), smoothed_p, "Processed"

# ── Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(title="VideoMAE RTSP App") as demo:
    #sensitivity = gr.Slider(label="Sensitivity", minimum=0.0, maximum=1.0, step=0.01, value=0.2)
    #stream_fps = gr.Slider(label="Stream FPS", minimum=5, maximum=15, step=1, value=STREAM_FPS)
    with gr.Row():
        # Left side: Logo and Title
        with gr.Column(scale=1, min_width=300):
            with gr.Row():
                gr.Image(
                    "https://myteam.ai/wp-content/uploads/2022/01/Plan-de-travail-3.png",
                    label="Logo",
                    show_label=False,
                    width=80,
                    height=80,
                    elem_classes=["no-border"]
                )
                gr.Markdown("**Violence Detection**", elem_classes=["title"])
        # Spacer to push sliders to the right
        gr.Column(scale=2)
        # Right side: Sliders
        with gr.Column(scale=1, min_width=300):
            sensitivity = gr.Slider(label="Sensitivity", minimum=0.0, maximum=1.0, step=0.01, value=0.2)
            stream_fps = gr.Slider(label="Stream FPS", minimum=5, maximum=15, step=1, value=STREAM_FPS)

    with gr.Tab("Video File"):
        file_in = gr.Video(format="mp4")
        vid_out = gr.Video()
        plot_out = gr.Image()
        beep_out = gr.Audio()
        gr.Button("Analyze").click(fn=run_pipeline, inputs=[file_in, sensitivity], outputs=[vid_out, plot_out, beep_out])

    with gr.Tab("Live Stream"):
        cam_in = gr.Image(sources='webcam', streaming=True, type='numpy')
        cam_out = gr.Image(streaming=True)
        cam_plot = gr.Plot()
        cam_prob = gr.Number()
        cam_status = gr.Textbox()
        cam_state = gr.State(value=deque(maxlen=CLIP))
        cam_hist = gr.State(value=[])
        cam_in.stream(fn=live_handler, inputs=[cam_in, cam_state, cam_hist, sensitivity, stream_fps],
                      outputs=[cam_state, cam_out, cam_hist, cam_plot, cam_prob, cam_status])

    with gr.Tab("HTTP Stream"):
        gr.Markdown(
            "Enter two HTTP stream URLs (e.g., ngrok URLs or http://192.168.1.18:8080/video). Border & plot turn red when `prob > sensitivity`."
        )
        with gr.Row():
            with gr.Column():
                http_url = gr.Textbox(label="HTTP URL 1", value="https://a-90-90-61-176.ngrok-free.app/video")
                http_out = gr.Image(streaming=True, label='Annotated HTTP Feed 1')
                http_plot = gr.Plot(label='Rolling Fight Probability (HTTP 1)')
                http_prob = gr.Number(label='Current Probability (HTTP 1)', precision=2)
                http_status = gr.Textbox(label="Processing Status (HTTP 1)")
            with gr.Column():
                http_url2 = gr.Textbox(label="HTTP URL 2", value="https://b-90-90-61-177.ngrok-free.app/video")
                http_out2 = gr.Image(streaming=True, label='Annotated HTTP Feed 2')
                http_plot2 = gr.Plot(label='Rolling Fight Probability (HTTP 2)')
                http_prob2 = gr.Number(label='Current Probability (HTTP 2)', precision=2)
                http_status2 = gr.Textbox(label="Processing Status (HTTP 2)")

        http_frame_state = gr.State(value=deque(maxlen=CLIP))
        http_prob_state = gr.State(value=[])
        http_frame_state2 = gr.State(value=deque(maxlen=CLIP))
        http_prob_state2 = gr.State(value=[])

        http_url.change(
            fn=http_stream_handler,
            inputs=[http_url, http_frame_state, http_prob_state, sensitivity, stream_fps],
            outputs=[http_frame_state, http_out, http_prob_state, http_plot, http_prob, http_status]
        )
        http_url2.change(
            fn=http_stream_handler2,
            inputs=[http_url2, http_frame_state2, http_prob_state2, sensitivity, stream_fps],
            outputs=[http_frame_state2, http_out2, http_prob_state2, http_plot2, http_prob2, http_status2]
        )

        timer = gr.Timer(value=1/STREAM_FPS)
        timer.tick(
            fn=http_stream_handler,
            inputs=[http_url, http_frame_state, http_prob_state, sensitivity, stream_fps],
            outputs=[http_frame_state, http_out, http_prob_state, http_plot, http_prob, http_status]
        )
        timer2 = gr.Timer(value=1/STREAM_FPS)
        timer2.tick(
            fn=http_stream_handler2,
            inputs=[http_url2, http_frame_state2, http_prob_state2, sensitivity, stream_fps],
            outputs=[http_frame_state2, http_out2, http_prob_state2, http_plot2, http_prob2, http_status2]
        )

demo.queue().launch(server_name="127.0.0.1", server_port = 7860, share=False)