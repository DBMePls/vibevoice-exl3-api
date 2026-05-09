import os
import sys
import time
import torch
import numpy as np
from collections import defaultdict

import vibevoice_api.config as config_mod
import vibevoice_api.tts_engine as engine_mod

DIFF_PATH = "./vibevoice/models--tensorbanana--vibevoice-7b-no-llm-bf16/snapshots/994ab69dbaca4e0a1ef2078cdc54117fdef99055/"
LLM_PATH  = "./vibevoice/models--tensorbanana--vibevoice-7b-exl3-8bit/"

config_mod.CONFIG = config_mod.ServerConfig(
    diffusion_model_path=DIFF_PATH,
    llm_model_path=LLM_PATH,
    quantization_mode="bf16",
    attention_type="sdpa",   # or whatever you normally use
    ddpm_steps=20
)

# Make sure tts_engine sees the same config object
engine_mod.CONFIG = config_mod.CONFIG

# --- CONFIGURATION (Adjust these if needed, or rely on defaults) ---
# We force these to ensure consistent profiling
os.environ["VIBEVOICE_DDPM_STEPS"] = "20" 
os.environ["VIBEVOICE_USE_SAMPLING"] = "0"

# Import VibeVoice modules
# Ensure we are in the root directory
sys.path.insert(0, os.getcwd())
try:
    from vibevoice_api import tts_engine, config
    from vvembed.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
except ImportError:
    print("Error: Run this script from the VibeVoice-API root directory.")
    sys.exit(1)

# --- THE PROFILER ENGINE ---

class Profiler:
    def __init__(self):
        self.events = defaultdict(list)
        self.enabled = False

    def reset(self):
        self.events = defaultdict(list)

    def start_capture(self):
        self.enabled = True
        torch.cuda.synchronize()

    def stop_capture(self):
        self.enabled = False
        torch.cuda.synchronize()

    def record(self, name):
        if not self.enabled:
            return NopContext()
        return ProfileContext(name, self.events)

class NopContext:
    def __enter__(self): pass
    def __exit__(self, *args): pass

class ProfileContext:
    def __init__(self, name, storage):
        self.name = name
        self.storage = storage
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def __enter__(self):
        # CPU Start Time
        self.cpu_t0 = time.perf_counter()
        # GPU Start Marker
        self.start_event.record()
    
    def __exit__(self, *args):
        # GPU End Marker
        self.end_event.record()
        # CPU End Time
        self.cpu_t1 = time.perf_counter()
        
        self.storage[self.name].append({
            "cpu_start": self.cpu_t0,
            "cpu_end": self.cpu_t1,
            "gpu_start_evt": self.start_event,
            "gpu_end_evt": self.end_event
        })

PROFILER = Profiler()

# --- MONKEY PATCH THE MODEL ---

# We replace the actual method in the class definition with our instrumented version
original_sample_method = VibeVoiceForConditionalGenerationInference.sample_speech_tokens

def profiled_sample_speech_tokens(self, condition, neg_condition, cfg_scale=1.3, increase_cfg=False):
    # Only profile if globally enabled
    if not PROFILER.enabled:
        return original_sample_method(self, condition, neg_condition, cfg_scale, increase_cfg)

    print("\n--- ENTERING DIFFUSION LOOP (PROFILING) ---")
    
    with PROFILER.record("0_Setup"):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        batch_size = condition.shape[0]
        device = self.model.prediction_head.device
        dtype = self.model.prediction_head.dtype
        
        conditions = torch.cat([condition, neg_condition], dim=0).to(device=device, dtype=dtype)
        speech = torch.randn((batch_size, self.config.acoustic_vae_dim), device=device, dtype=dtype)
        timesteps = self.model.noise_scheduler.timesteps.to(device)

    total_steps = len(timesteps)
    
    # Pre-allocate if possible to test 'Raw Math' vs 'Alloc Overhead'
    # But let's profile the 'Standard' path first to find the culprit.

    for i, t in enumerate(timesteps):
        step_name = f"Step_{i}"
        
        # 1. Dynamic CFG Calc
        with PROFILER.record("1_CFG_Calc_CPU"):
            if increase_cfg:
                progress = i / total_steps
                current_scale = cfg_scale * (1.0 + 0.5 * (progress < 0.5))
            else:
                current_scale = cfg_scale

        # 2. Input Concatenation (Memory Alloc?)
        with PROFILER.record("2_Input_Concat"):
            latent_input = torch.cat([speech] * 2)
            t_input = t.repeat(latent_input.shape[0]).to(device).to(dtype)

        # 3. The Big One: Model Forward
        with PROFILER.record("3_Model_Forward"):
            noise_pred = self.model.prediction_head(latent_input, t_input, condition=conditions)

        # 4. CFG Split & Math
        with PROFILER.record("4_CFG_Apply"):
            cond_pred, uncond_pred = noise_pred.chunk(2)
            guided_pred = uncond_pred + current_scale * (cond_pred - uncond_pred)

        # 5. Scheduler Step (Python Logic Overhead?)
        with PROFILER.record("5_Scheduler_Step"):
            speech = self.model.noise_scheduler.step(guided_pred, t, speech).prev_sample

    print("--- EXITING DIFFUSION LOOP ---")
    return speech

# Apply Patch
VibeVoiceForConditionalGenerationInference.sample_speech_tokens = profiled_sample_speech_tokens

# --- ANALYSIS REPORT ---

def print_analysis():
    torch.cuda.synchronize() # Ensure everything finished
    
    print("\n" + "="*80)
    print(f"{'OPERATION':<25} | {'CPU LAUNCH (ms)':<15} | {'GPU EXEC (ms)':<15} | {'RATIO (G/C)':<10}")
    print("="*80)
    
    total_cpu_accum = 0
    total_gpu_accum = 0
    
    # Aggregate stats
    stats_summary = {}
    
    for op_name, runs in PROFILER.events.items():
        if not runs: continue
        
        cpu_times = []
        gpu_times = []
        
        for r in runs:
            cpu_ms = (r["cpu_end"] - r["cpu_start"]) * 1000
            # GPU time requires synchronization of events
            gpu_ms = r["gpu_start_evt"].elapsed_time(r["gpu_end_evt"])
            
            cpu_times.append(cpu_ms)
            gpu_times.append(gpu_ms)
            
        avg_cpu = np.mean(cpu_times)
        avg_gpu = np.mean(gpu_times)
        total = np.sum(gpu_times)
        
        stats_summary[op_name] = (avg_cpu, avg_gpu, len(runs))
        
        ratio = avg_gpu / avg_cpu if avg_cpu > 0 else 0
        ratio_str = f"{ratio:.2f}x"
        
        print(f"{op_name:<25} | {avg_cpu:15.4f} | {avg_gpu:15.4f} | {ratio_str:<10}")
        
        # Approximate loop totals (ignoring setup)
        if op_name.startswith(("1_", "2_", "3_", "4_", "5_")):
            total_cpu_accum += avg_cpu
            total_gpu_accum += avg_gpu

    print("-" * 80)
    print(f"{'LOOP TOTAL (Per Step)':<25} | {total_cpu_accum:15.4f} | {total_gpu_accum:15.4f}")
    
    # Calculating the "Gap"
    # Gap is time spent where CPU is working but GPU is waiting for the next command
    # Roughly: Max(CPU, GPU) - GPU is "Idle" time if pipelining works perfectly.
    # But in blocked scenarios, it's additive.
    
    print("\n--- DIAGNOSIS ---")
    if total_cpu_accum > total_gpu_accum:
        print(f"CRITICAL: CPU is slower than GPU by {(total_cpu_accum - total_gpu_accum):.2f}ms per step.")
        print("Verdict: CPU BOUND. The Celeron cannot feed the 3090 fast enough.")
    else:
        print(f"GPU is slower than CPU. Overhead is likely PCIe transfers or hidden synchronization.")
        print("Verdict: GPU/PCIe BOUND.")

# --- MAIN EXECUTION ---

def main():
    print(">> Loading VibeVoice Model (This may take time)...")
    
    # Load model using standard API logic
    # This respects the CLI args passed to this script if any, or env vars
    # We force some defaults for consistency
    engine = tts_engine._load_model(
        config_mod.CONFIG.diffusion_model_path,
        config_mod.CONFIG.llm_model_path,
        config_mod.CONFIG.quantization_mode,
        config_mod.CONFIG.attention_type
    )
    
    print("\n>> Warming up...")
    # Run one pass without profiling to compile kernels/allocate cache
    tts_engine.synthesize(
        root_dir=os.getcwd(),
        text="Warmup text.",
        voice="Alice",
        ddpm_steps=5 # Short warmup
    )
    
    print("\n>> STARTING PROFILE RUN...")
    PROFILER.reset()
    PROFILER.start_capture()
    
    start_time = time.perf_counter()
    tts_engine.synthesize(
        root_dir=os.getcwd(),
        text="Profiling the diffusion loop execution time.",
        voice="Alice",
        ddpm_steps=20 # Standard length
    )
    end_time = time.perf_counter()
    
    PROFILER.stop_capture()
    
    print(f"\n>> Total Generation Wall Clock: {(end_time - start_time):.4f}s")
    print_analysis()

if __name__ == "__main__":
    main()