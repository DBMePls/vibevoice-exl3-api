# --- START OF FILE diagnostic_generation.py ---
import os
import sys
import time
import torch
import numpy as np
import subprocess
import argparse
import re

# 1. Load .env BEFORE anything else so default paths match the server
def _load_dotenv():
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path): return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", s)
            if not m: continue
            k, v = m.group(1), m.group(2)
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            os.environ.setdefault(k, v)

_load_dotenv()

# Ensure we are in the root directory
sys.path.insert(0, os.getcwd())

from vibevoice_api import tts_engine
from vibevoice_api.config import CONFIG
from exllamav3 import Cache
from exllamav3.tokenizer import MMEmbedding

# --- CACHE HELPERS ---
def _create_cache(model, max_num_tokens):
    cache = Cache(model, max_num_tokens=max_num_tokens)
    for module in model.get_cache_layers():
        layer = cache.layers[module.layer_idx]
        layer.alloc(module.device)
    return cache

def _destroy_cache(cache, model):
    for module in model.get_cache_layers():
        layer = cache.layers[module.layer_idx]
        layer.free()
    cache.detach_from_model(model)

# --- DIAGNOSTIC HELPERS ---
def print_header(title):
    print(f"\n{'='*85}")
    print(f"{title:^85}")
    print(f"{'='*85}")

def load_audio_ffmpeg_strict(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ar", "24000", "-ac", "1", "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    
    # Strict -25 dBFS Normalization
    target_lin = 10 ** (-25.0 / 20.0)
    wav = wav * (target_lin / (np.sqrt(np.mean(wav**2)) + 1e-6))
    maxabs = np.max(np.abs(wav))
    if maxabs > 1.0: wav /= (maxabs + 1e-6)
    return wav

# --- MAIN DEEP PROFILER ---
def run_debugger(args):
    print_header("VIBEVOICE HARDWARE BOTTLENECK PROFILER")

    # Override config with args if provided
    if args.diffusion_model_path:
        CONFIG.diffusion_model_path = args.diffusion_model_path
    if args.llm_model_path:
        CONFIG.llm_model_path = args.llm_model_path

    print(f"[INFO] Target LLM: {CONFIG.llm_model_path}")
    print(f"[INFO] Target DiT: {CONFIG.diffusion_model_path}")
    print("[INFO] Loading engine... (This takes a few seconds)")
    
    t0 = time.perf_counter()
    engine = tts_engine._get_engine()
    device = engine.model.output_device or "cuda:0"
    print(f"[TIME] Engine loaded successfully in {(time.perf_counter()-t0):.2f} seconds.")

    text = "This is a hardware profiler test. We are tracking exact CPU and GPU timing to eliminate all bottlenecks."
    
    voice_path = os.path.join(os.getcwd(), "demo", "voices", "en-Alice_woman.wav")
    if not os.path.exists(voice_path):
        print(f"[ERROR] Could not find voice file at {voice_path}. Make sure you are in the root directory.")
        return

    wav_norm = load_audio_ffmpeg_strict(voice_path)
    
    with torch.inference_mode():
        audio_tensor = torch.from_numpy(wav_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        voice_embeddings = engine.model.worker.encode_acoustic(audio_tensor).cpu()
    
    token_string = torch.full((1, voice_embeddings.shape[1]), -1, dtype=torch.long)
    voice_mme = MMEmbedding(embeddings=voice_embeddings.squeeze(0).half(), token_string=token_string, text_alias="<$VOICE$>")

    prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
    prompt += " Voice input:\n Speaker 0:<|vision_start|><$VOICE$><|vision_end|>\n"
    prompt += f" Text input:\n Speaker 0: {text.strip()}\n Speech output:\n<|vision_start|>"
    
    input_ids = engine.tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True, embeddings=[voice_mme])
    
    print_header("STARTING HARDWARE PROFILER")
    
    cfg = 1.3
    use_cfg = True
    seed = int(time.perf_counter())
    
    # PROFILING VARIABLES
    total_cpu_queue_time = 0.0
    total_pcie_sync_time = 0.0
    cuda_events_start = []
    cuda_events_end = []
    
    with torch.inference_mode():
        if use_cfg:
            neg_input_ids = torch.tensor([[engine.speech_start_id]], dtype=torch.long, device="cpu")
            inputs_embeds_neg = engine.model.modules[0].forward(neg_input_ids, {})
            _, neg_hidden = engine.model.forward(inputs_embeds=inputs_embeds_neg, params={"attn_mode": "flash_attn_nc"})
            cond_neg = neg_hidden[:, -1:, :].half()
        else:
            cond_neg = None
            
        cache_pos = _create_cache(engine.model, max_num_tokens=8192)

        try:
            # Prefill
            inputs_embeds_pos = engine.model.modules[0].forward(input_ids, {"indexed_embeddings": [voice_mme]})
            params_pos = {"attn_mode": "flash_attn", "cache": cache_pos, "past_len": 0, "batch_shape": (1, 8192)}
            logits_pos, hidden_last_pos = engine.model.forward(inputs_embeds=inputs_embeds_pos, params=params_pos)
            
            past_len = inputs_embeds_pos.shape[1]
            all_latents = []
            chunk_size = 30
            eos_flag = torch.zeros(1, dtype=torch.bool, device=device)
            tokens_gen = 0
            
            print(f"[INFO] AR Pipelined Loop running with chunk size: {chunk_size}")
            
            sys.stdout.write("Generating: [")
            sys.stdout.flush()
            
            wall_start = time.perf_counter()
            
            for chunk_start in range(0, 1500, chunk_size):
                chunk_latents = []
                chunk_preds = []
                
                # --- START CPU QUEUE TIMER ---
                t_cpu_start = time.perf_counter()
                
                # --- START CUDA HARDWARE TIMER ---
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_start.record()
                
                for t in range(chunk_start, min(chunk_start + chunk_size, 1500)):
                    cond_pos = hidden_last_pos[:, -1:, :].half()
                    
                    z = engine.model.worker.sample_latent(cond_pos, cond_neg if use_cfg else cond_pos, cfg, seed + t, False)
                    chunk_latents.append(z)
                    step_embed = engine.model.worker.acoustic_connector_forward(z.squeeze(1)).unsqueeze(1)
                    
                    params_pos = {"attn_mode": "flash_attn", "cache": cache_pos, "past_len": past_len, "batch_shape": (1, 8192)}
                    logits_pos, hidden_last_pos = engine.model.forward(inputs_embeds=step_embed.to(inputs_embeds_pos.dtype), params=params_pos)
                    
                    past_len += 1
                    tokens_gen += 1
                    
                    pred_id = logits_pos[0, -1, :].argmax()
                    chunk_preds.append(pred_id)
                    eos_flag.logical_or_(pred_id == engine.speech_end_id)
                
                # --- STOP CUDA HARDWARE TIMER ---
                ev_end.record()
                cuda_events_start.append(ev_start)
                cuda_events_end.append(ev_end)
                
                # --- STOP CPU QUEUE TIMER ---
                t_cpu_end = time.perf_counter()
                total_cpu_queue_time += (t_cpu_end - t_cpu_start)
                
                # Asynchronous Progress Bar (No sync overhead!)
                sys.stdout.write("#")
                sys.stdout.flush()
                
                # --- START SYNC TIMER ---
                t_sync_start = time.perf_counter()
                is_eos = eos_flag.item() # <--- THE ONLY SYNC
                total_pcie_sync_time += (time.perf_counter() - t_sync_start)
                # --- STOP SYNC TIMER ---
                
                all_latents.extend(chunk_latents)
                
                if is_eos:
                    preds_cpu = torch.stack(chunk_preds).cpu()
                    eos_indices = (preds_cpu == engine.speech_end_id).nonzero(as_tuple=True)[0]
                    if len(eos_indices) > 0:
                        first_eos_idx = eos_indices[0].item()
                        trim_count = len(chunk_preds) - first_eos_idx - 1
                        if trim_count > 0:
                            all_latents = all_latents[:-trim_count]
                            tokens_gen -= trim_count
                    break
                    
            wall_end = time.perf_counter()
            sys.stdout.write("] Done!\n")
            
        finally:
            _destroy_cache(cache_pos, engine.model)

        latents = torch.cat(all_latents, dim=1)
        audio_tensor = engine.model.worker.decode_vae(latents)
        wav = audio_tensor.cpu().numpy()

    import soundfile as sf
    sf.write("debugger_output_cpp.wav", wav, 24000)
    
    # Calculate True GPU Hardware Time
    torch.cuda.synchronize() # Final sync to ensure events are recorded
    total_gpu_hardware_time = sum(s.elapsed_time(e) for s, e in zip(cuda_events_start, cuda_events_end)) / 1000.0
    
    total_wall_time = wall_end - wall_start
    audio_duration = len(wav) / 24000
    rtf = audio_duration / total_wall_time
    gpu_utilization = (total_gpu_hardware_time / total_wall_time) * 100

    print_header("PROFILING RESULTS")
    print(f"Total Wall Clock Time:      {total_wall_time:.4f} seconds")
    print(f"Audio Duration Generated:   {audio_duration:.4f} seconds")
    print(f"Real-Time Factor (RTF):     {rtf:.2f}x (Higher is better)")
    print(f"Frames Per Second (FPS):    {(tokens_gen / total_wall_time):.1f}")
    print("-" * 50)
    print(f"CPU Python Queue Time:      {total_cpu_queue_time:.4f} sec ({(total_cpu_queue_time/total_wall_time)*100:.1f}% of wall)")
    print(f"PCIe Sync & Idle Time:      {total_pcie_sync_time:.4f} sec ({(total_pcie_sync_time/total_wall_time)*100:.1f}% of wall)")
    print(f"True GPU Compute Time:      {total_gpu_hardware_time:.4f} sec")
    print("-" * 50)
    print(f">>> GPU UTILIZATION:        {gpu_utilization:.1f}% <<<")
    
    if gpu_utilization > 85:
        print("\n[CONCLUSION] PERFECT. Your CPU is no longer a bottleneck. The RTX 3090 is running at max speed.")
    else:
        print("\n[CONCLUSION] Sub-optimal. Consider increasing chunk_size to 50 to hide more PCIe latency.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model_path", type=str, default=None, help="Override LLM path")
    parser.add_argument("--diffusion_model_path", type=str, default=None, help="Override DiT path")
    args = parser.parse_args()

    try:
        run_debugger(args)
    except Exception as e:
        import traceback
        print("\n[CRITICAL ERROR] The diagnostic script crashed!")
        traceback.print_exc()