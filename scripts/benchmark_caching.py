#!/usr/bin/env python3
import os
import time
import io
import wave
from openai import OpenAI

# 1. Initialize OpenAI Client pointing to your local server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1", 
    api_key="sk-test" # Auth is disabled by default, but client requires a string
)

# 2. Define the test paragraph (long enough to show clear differences)
TEST_TEXT = (
    "This is a comprehensive performance test of the Vibe Voice system using the optimized "
    "ExLlama V3 backend. We are evaluating the impact of different negative caching strategies "
    "on generation speed. By using a longer paragraph, we can clearly observe how skipping "
    "the negative language model steps reduces GPU memory bandwidth overhead and improves the "
    "Real-Time Factor."
)

MODES_TO_TEST = [
    ("Every Step (0)", 0),        # Baseline: Slowest, evaluates negative prompt every step
    ("Cache Every 2 Steps", 2),   # Balanced: Skips 2 steps
    ("Cache Every 4 Steps", 4),   # Fast: Skips 4 steps
    ("Static (-1)", -1),          # Fastest: Evaluates negative prompt only once at the beginning
]

def get_wav_duration(audio_bytes: bytes) -> float:
    """Reads the WAV header to get exact duration in seconds."""
    with wave.open(io.BytesIO(audio_data), 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def main():
    os.makedirs("outputs/benchmark", exist_ok=True)
    
    print("="*80)
    print("\U0001f525 VibeVoice Caching Performance Benchmark \U0001f525")
    print("="*80)
    
    # --- WARM-UP ---
    print("\n[1/3] Running warm-up generation (to clear CUDA init overhead)...")
    client.audio.speech.create(
        model="vibevoice", voice="Alice", input="Warm up.", response_format="wav"
    )
    print("Warm-up complete.\n")
    
    # --- BENCHMARK ---
    print(f"[2/3] Running Benchmarks (Text Length: {len(TEST_TEXT)} chars)...")
    results = []
    
    for name, cache_val in MODES_TO_TEST:
        print(f"  -> Testing: {name} ... ", end="", flush=True)
        
        # Pass the caching parameter via extra_body
        extra_body = {"negative_llm_steps_to_cache": cache_val}
        
        t0 = time.perf_counter()
        
        response = client.audio.speech.create(
            model="vibevoice", 
            voice="Alice",
            input=TEST_TEXT,
            response_format="wav",
            extra_body=extra_body,
            speed=1.0 # Force 1.0 speed for fair comparison
        )
        
        global audio_data
        audio_data = response.read()
        t1 = time.perf_counter()
        
        # Calculate metrics
        latency = t1 - t0
        duration = get_wav_duration(audio_data)
        rtf = duration / latency if latency > 0 else 0.0
        
        out_path = f"outputs/benchmark/mode_{cache_val}.wav"
        with open(out_path, "wb") as f:
            f.write(audio_data)
            
        print(f"Done! ({latency:.2f}s)")
        
        results.append({
            "mode": name,
            "latency": latency,
            "duration": duration,
            "rtf": rtf,
            "file": out_path
        })

    # --- RESULTS ---
    print("\n[3/3] Benchmark Results:")
    print("="*90)
    print(f"{'Mode':<25} | {'Latency (s)':<12} | {'Audio Dur (s)':<15} | {'RTF':<8} | {'Speedup':<8}")
    print("-" * 90)
    
    # Baseline is "Every Step (0)"
    baseline_latency = next(r["latency"] for r in results if r["mode"] == "Every Step (0)")
    
    for r in results:
        speedup = baseline_latency / r["latency"]
        print(f"{r['mode']:<25} | {r['latency']:<12.2f} | {r['duration']:<15.2f} | {r['rtf']:<8.2f}x | {speedup:<8.2f}x")
    
    print("="*90)
    print("\n\u2705 Audio files saved to outputs/benchmark/ for quality comparison.")
    print("Note: RTF > 1.0 means it generates faster than real-time playback.")

if __name__ == "__main__":
    main()