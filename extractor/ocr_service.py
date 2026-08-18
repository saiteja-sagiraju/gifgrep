import sys
import re
import numpy as np
import easyocr
from PIL import Image
from typing import List

# Fix Windows console UTF-8 output encoding for progress bars and special characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

class EasyOCRService:
    def __init__(self, languages: List[str] = ['en']):
        # Explicitly disable GPU to keep VRAM completely dedicated to VideoCLIP
        self.reader = easyocr.Reader(languages, gpu=False, verbose=False)

    def extract_text(self, pil_frames: List[Image.Image]) -> str:
        """Extracts and deduplicates burned-in text across multiple GIF frames."""
        seen_phrases = set()
        deduped_tokens: List[str] = []

        for frame in pil_frames:
            frame_np = np.array(frame)
            results = self.reader.readtext(frame_np, detail=0)

            for line in results:
                # Handle possible variations in EasyOCR return types
                if isinstance(line, str):
                    cleaned = line.strip()
                elif isinstance(line, (list, tuple)) and len(line) >= 2:
                    cleaned = str(line[1]).strip()
                elif isinstance(line, dict):
                    cleaned = str(line.get("text", "")).strip()
                else:
                    cleaned = str(line).strip()

                # Normalize key for deduplication (lowercase, strip special chars)
                norm_key = re.sub(r'\W+', '', cleaned.lower())
                
                if norm_key and len(norm_key) > 1 and norm_key not in seen_phrases:
                    seen_phrases.add(norm_key)
                    deduped_tokens.append(cleaned)

        return " ".join(deduped_tokens)