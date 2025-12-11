# CLIP/BLIP-based Vision-Language Encoder
"""
Vision-Language Encoder for AutoGuard-RL
Uses CLIP/BLIP models to compute semantic similarity between driving scenes
and textual safety prompts.
"""
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np


class ClipEncoder:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_image(self, image_np):
        """Converts np.ndarray or PIL image into normalized CLIP embedding."""
        if isinstance(image_np, np.ndarray):
            image = Image.fromarray(image_np)
        else:
            image = image_np
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        return img_emb.cpu()

    def encode_texts(self, texts):
        """Encodes textual safety prompts into CLIP embeddings."""
        inputs = self.processor(text=texts, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            txt_emb = self.model.get_text_features(**inputs)
            txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        return txt_emb.cpu()

    def safety_score(self, image_np, unsafe_prompts=None):
        """Estimates a safety risk [0,1] based on similarity to unsafe prompts."""
        if unsafe_prompts is None:
            unsafe_prompts = [
                "pedestrian on the road",
                "red traffic light",
                "collision ahead",
                "car too close",
                "driving on wrong lane"
            ]
        img_e = self.encode_image(image_np)
        txt_e = self.encode_texts(unsafe_prompts)
        sims = (img_e @ txt_e.T).squeeze(0).numpy()
        max_sim = float(sims.max())
        risk = (max_sim + 1) / 2  # map [-1,1] -> [0,1]
        return risk
