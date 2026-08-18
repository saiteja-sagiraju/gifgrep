# extractor/pipeline.py
import threading
from typing import Dict, Any, Union
from extractor.videoclip_service import VideoCLIPService
from extractor.ocr_service import EasyOCRService

class GIFExtractionPipeline:
    def __init__(self):
        self.videoclip = VideoCLIPService()
        self.ocr = EasyOCRService()
        # Thread lock strictly for GPU operations
        self.gpu_lock = threading.Lock()

    def process_gif(self, gif_path_or_bytes: Union[str, bytes]) -> Dict[str, Any]:
        """
        Executes the dual-modality extraction:
        1. Frame sampling (CPU)
        2. EasyOCR text extraction (CPU)
        3. VideoCLIP temporal embedding (GPU, protected by thread lock)
        """
        # Step 1: Extract 8 frames
        frames = self.videoclip.extract_frames_from_gif(gif_path_or_bytes, num_frames=8)

        # Step 2: CPU OCR (Runs concurrently across worker threads without lock)
        ocr_text = self.ocr.extract_text(frames)

        # Step 3: GPU Embedding (Protected by GPU lock to prevent VRAM spikes / OOM)
        with self.gpu_lock:
            embedding = self.videoclip.embed_frames(frames)

        return {
            "ocr_text": ocr_text,
            "embedding": embedding
        }