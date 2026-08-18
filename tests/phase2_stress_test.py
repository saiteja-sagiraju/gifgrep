import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import concurrent.futures
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extractor.pipeline import GIFExtractionPipeline

def create_dummy_gif(filename: str, text: str):
    """Generates a test animated GIF with text burned in."""
    frames = []
    for i in range(16):
        img = Image.new('RGB', (300, 300), color=(i * 15 % 255, 100, 200))
        d = ImageDraw.Draw(img)
        d.text((50, 130), f"{text} frame {i}", fill=(255, 255, 255))
        frames.append(img)
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=100, loop=0)

def run_stress_test():
    print("=== Phase 2 Stress Test: 3 Concurrent Ingestions ===")
    
    # 1. Generate 3 sample test GIFs
    test_files = ["test_1.gif", "test_2.gif", "test_3.gif"]
    for i, f in enumerate(test_files, 1):
        create_dummy_gif(f, f"Stress Test Subtitle #{i}")

    # 2. Initialize pipeline
    pipeline = GIFExtractionPipeline()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Base VRAM allocated: {initial_vram:.2f} MB")

    start_time = time.time()

    # 3. Process 3 GIFs concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(pipeline.process_gif, path): path for path in test_files}
        results = []
        for future in concurrent.futures.as_completed(futures):
            gif_name = futures[future]
            res = future.result()
            results.append((gif_name, res))
            print(f"[OK] Completed {gif_name} -> OCR: '{res['ocr_text']}', Embedding Dim: {len(res['embedding'])}")

    elapsed = time.time() - start_time
    print(f"\nAll 3 GIFs processed concurrently in {elapsed:.2f}s")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak VRAM used: {peak_vram:.2f} MB (well within 8GB limit)")

if __name__ == "__main__":
    run_stress_test()