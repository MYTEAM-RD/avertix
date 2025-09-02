from transformers import VideoMAEForVideoClassification, AutoImageProcessor

def load_model(model_path: str = "model/"):
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    return model, processor
