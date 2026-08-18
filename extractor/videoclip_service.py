import io
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageSequence
from typing import List, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VideoCLIP-XL")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modeling import VideoCLIP_XL
from utils.text_encoder import text_encoder
from PIL.Image import Resampling

BICUBIC = Resampling.BICUBIC

# Normalization constants (ImageNet mean & std)
V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
V_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

class VideoCLIPService:
    def __init__(self, weights_path: str | None = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading VideoCLIP-XL onto {self.device}...")
        
        if weights_path is None:
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "VideoCLIP-XL.bin"),
                os.path.join(os.path.dirname(__file__), "..", "VideoCLIP-XL", "VideoCLIP-XL.bin"),
                "extractor/VideoCLIP-XL.bin",
                "VideoCLIP-XL/VideoCLIP-XL.bin",
                "VideoCLIP-XL.bin",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    weights_path = p
                    break
            if weights_path is None:
                weights_path = "extractor/VideoCLIP-XL.bin"

        self.model = VideoCLIP_XL()
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        
        if self.device.type == "cuda":
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"VideoCLIP-XL loaded successfully into VRAM: {allocated_mb:.2f} MB allocated ({reserved_mb:.2f} MB reserved)")

    @staticmethod
    def extract_frames_from_gif(gif_path_or_bytes: Union[str, bytes], num_frames: int = 8) -> List[Image.Image]:
        """Extracts evenly spaced RGB frames from an animated GIF."""
        gif_input = io.BytesIO(gif_path_or_bytes) if isinstance(gif_path_or_bytes, bytes) else gif_path_or_bytes
        with Image.open(gif_input) as img:
            frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(img)]
        
        if not frames:
            raise ValueError("No frames found in GIF")

        total_frames = len(frames)
        if total_frames <= num_frames:
            step = 1
            sampled = frames[:]
            while len(sampled) < num_frames:
                sampled.append(sampled[-1])
        else:
            step = total_frames / num_frames
            sampled = [frames[int(i * step)] for i in range(num_frames)]
            
        return sampled[:num_frames]

    def preprocess_frames(self, pil_frames: List[Image.Image]) -> torch.Tensor:
        """Resizes, normalizes, and shapes frames into (1, T, 3, 224, 224) tensor."""
        vid_tube = []
        for img in pil_frames:
            img_resized = img.resize((224, 224), BICUBIC)
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            arr = (arr - V_MEAN) / V_STD
            arr = np.expand_dims(arr, axis=(0, 1))  # (1, 1, 224, 224, 3)
            vid_tube.append(arr)
            
        vid_tube = np.concatenate(vid_tube, axis=1)  # (1, T, 224, 224, 3)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))  # (1, T, 3, 224, 224)
        return torch.from_numpy(vid_tube).float().to(self.device)

    @torch.no_grad()
    def embed_frames(self, pil_frames: List[Image.Image]) -> List[float]:
        """Generates L2-normalized 768-dim temporal embedding from PIL frames."""
        tensor_input = self.preprocess_frames(pil_frames)
        features = self.model.vision_model.get_vid_features(tensor_input).float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().squeeze(0).numpy().tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> List[float]:
        """Generates L2-normalized 768-dim text embedding."""
        text_tokens = text_encoder.tokenize([text], truncate=True).to(self.device)
        features = self.model.text_model.encode_text(text_tokens).float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().squeeze(0).numpy().tolist()