import torch
import cv2
import numpy as np
from transformers import VideoMAEForVideoClassification, AutoImageProcessor

class ViolenceDetector:
    def __init__(self, model_dir="../models", device='cuda'):
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_dir).to(device)
        self.model.eval()
        self.device = device

    def predict_clip(self, frames):
        inputs = self.processor(frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs[1]  # probability of Danger class
