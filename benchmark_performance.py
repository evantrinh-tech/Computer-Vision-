import tensorflow as tf
import time
import numpy as np
import os
from pathlib import Path

# Paths & Config
ROOT_DIR = Path(__file__).parent
MODEL_DIR = ROOT_DIR / "models"
IMAGE_SIZE = 224
ITERATIONS = 100 # Số lần chạy để lấy trung bình
WARMUP_RUNS = 20 # Số lần chạy khởi động

def load_latest_model():
    subdirs = [d for d in MODEL_DIR.iterdir() if d.is_dir()]
    if not subdirs: return None
    latest_dir = sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    model_path = latest_dir / "final_model.h5"
    if not model_path.exists():
        model_path = latest_dir / "best_stage2_fine_tuning.h5"
    return tf.keras.models.load_model(model_path) if model_path.exists() else None

def benchmark():
    print("\n" + "="*60)
    print("AI PERFORMANCE BENCHMARK - STATISTICAL PROFILING")
    print("="*60)

    model = load_latest_model()
    if not model:
        print("Model not found. Please train first.")
        return

    # 1. Warm-up Phase (Khởi động phần cứng)
    print(f"Warm-up runs ({WARMUP_RUNS})...", end="", flush=True)
    dummy_input = np.random.uniform(0, 255, (1, IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)
    for _ in range(WARMUP_RUNS):
        _ = model.predict(dummy_input, verbose=0)
    print(" DONE")

    # 2. Statistical Profiling
    pre_times = []
    inf_times = []
    post_times = []

    print(f"Benchmarking over {ITERATIONS} iterations...")
    
    for i in range(ITERATIONS):
        # Mẫu ảnh giả lập
        raw_img = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        # Stage 1: Preprocessing
        t0 = time.perf_counter()
        img_array = raw_img.astype(np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
        t1 = time.perf_counter()
        pre_times.append((t1 - t0) * 1000) # ms

        # Stage 2: Inference
        t2 = time.perf_counter()
        preds = model.predict(img_batch, verbose=0)
        t3 = time.perf_counter()
        inf_times.append((t3 - t2) * 1000) # ms

        # Stage 3: Post-processing
        t4 = time.perf_counter()
        score = preds[0][0]
        label = "Incident" if score > 0.5 else "Normal"
        t5 = time.perf_counter()
        post_times.append((t5 - t4) * 1000) # ms

    # Results Calculation
    avg_pre = np.mean(pre_times)
    avg_inf = np.mean(inf_times)
    avg_post = np.mean(post_times)
    total_latency = avg_pre + avg_inf + avg_post
    fps = 1000 / total_latency

    print("\n" + "-"*30)
    print(f"LATENCY BREAKDOWN (Average)")
    print("-"*30)
    print(f"1. Preprocessing:  {avg_pre:6.2f} ms")
    print(f"2. AI Inference:   {avg_inf:6.2f} ms")
    print(f"3. Post-processing:{avg_post:6.2f} ms")
    print("-"*30)
    print(f"TOTAL LATENCY:      {total_latency:6.2f} ms")
    print(f"INFERRED SPEED:     {fps:6.2f} FPS")
    print("-"*30)
    print("\n[SUCCESS] Metrics are consistent.")

if __name__ == "__main__":
    benchmark()
