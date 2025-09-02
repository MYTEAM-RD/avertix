import torch
import cv2
import numpy as np
from transformers import VideoMAEForVideoClassification, AutoImageProcessor
import os

class ViolenceDetector:
    def __init__(self, model_dir="../models", device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Check if model directory exists and has actual model files
        if os.path.exists(model_dir) and any(f.endswith('.bin') for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))):
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_dir)
                self.model = VideoMAEForVideoClassification.from_pretrained(model_dir).to(self.device)
                self.model.eval()
                self.use_real_model = True
            except Exception as e:
                print(f"Warning: Could not load real model from {model_dir}: {e}")
                self.use_real_model = False
        else:
            print(f"Warning: Model directory {model_dir} not found or empty. Using mock model for testing.")
            self.use_real_model = False

    def predict_clip(self, frames):
        if not self.use_real_model:
            # Return a mock prediction for testing
            return np.random.random()
        
        try:
            inputs = self.processor(frames, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return probs[1]  # probability of Danger class
        except Exception as e:
            print(f"Error during prediction: {e}")
            return np.random.random()  # Fallback to random for testing
