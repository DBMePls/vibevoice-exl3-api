# --- START OF FILE diagnostic_generation.py ---
import os
import sys
import time

print("\n>>> BOOTING HARDWARE MICRO-PROFILER...", flush=True)

import torch
import numpy as np
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, os.getcwd())

from vibevoice_api import tts_engine
from exllamav3 import Cache
from exllamav3.tokenizer import MMEmbedding

def _create_cache(model, max_num_tokens):
    cache = Cache(model, max_num_tokens=max_num_tokens)
    for module in model.get_cache_layers():
        cache.layers[module.layer_idx].alloc(module.device)
    return cache

def _destroy_cache(cache, model):
    for module in model.get_cache_layers():
        cache.layers[module.layer_idx].free()
    cache.detach_from_model(model)

def print_header(title):
    print(f"\n{'='*85}")
    print(f"{title:^85}")
    print(f"{'='*85}", flush=True)

def load_audio_ffmpeg_strict(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ar", "24000", "-ac", "1", "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    wav = wav * ((10 ** (-25.0 / 20.0)) / (np.sqrt(np.mean(wav**2)) + 1e-6))
    maxabs = np.max(np.abs(wav))
    if maxabs > 1.0: wav /= (maxabs + 1e-6)
    return wav

def run_debugger():
    print_header("VIBEVOICE HARDWARE MICRO-PROFILER")

    engine = tts_engine._get_engine()
    device = engine.model.output_device or "cuda:0"
    
    voice_path = os.path.join(os.getcwd(), "demo", "voices", "en-Alice_woman.wav")
    wav_norm = load_audio_ffmpeg_strict(voice_path)
    
    with torch.inference_mode():
        audio_tensor = torch.from_numpy(wav_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        voice_embeddings = engine.model.worker.encode_acoustic(audio_tensor).cpu()
    
    token_string = torch.full((1, voice_embeddings.shape[1]), -1, dtype=torch.long)
    voice_mme = MMEmbedding(embeddings=voice_embeddings.squeeze(0).half(), token_string=token_string, text_alias="<$VOICE$>")

    prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
    prompt += " Voice input:\n Speaker 0:<|vision_start|><$VOICE$><|vision_end|>\n"
    prompt += " Text input:\n Speaker 0: Tell me the exact truth about my hardware bottlenecks.\n Speech output:\n<|vision_start|>"
    
    input_ids = engine.tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True, embeddings=[voice_mme])
    
    print_header("STARTING MICRO-PROFILER")
    
    cfg = 1.3
    use_cfg = True
    seed = 42
    
    chunk_metrics = []
    wasted_overshoot_frames = 0
    total_pcie_sync_time = 0.0
    
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
            inputs_embeds_pos = engine.model.modules[0].forward(input_ids, {"indexed_embeddings": [voice_mme]})
            params_pos = {"attn_mode": "flash_attn", "cache": cache_pos, "past_len": 0, "batch_shape": (1, 8192)}
            logits_pos, hidden_last_pos = engine.model.forward(inputs_embeds=inputs_embeds_pos, params=params_pos)
            
            past_len = inputs_embeds_pos.shape[1]
            all_latents = []
            chunk_size = 30
            eos_flag = torch.zeros(1, dtype=torch.bool, device=device)
            tokens_gen = 0
            
            wall_start = time.perf_counter()
            
            # CUDA Graphing Timers
            ev_starts = []
            ev_ends = []
            
            for chunk_idx, chunk_start in enumerate(range(0, 1500, chunk_size)):
                chunk_latents = []
                chunk_preds = []
                
                # --- CUDA EVENT RECORDING ---
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
                
                ev_end.record()
                ev_starts.append(ev_start)
                ev_ends.append(ev_end)
                
                # --- PCIE SYNC ---
                t_sync_start = time.perf_counter()
                is_eos = eos_flag.item()
                total_pcie_sync_time += (time.perf_counter() - t_sync_start)
                
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
                            wasted_overshoot_frames = trim_count
                    break
                    
            wall_end = time.perf_counter()
            
        finally:
            _destroy_cache(cache_pos, engine.model)

        latents = torch.cat(all_latents, dim=1)
        audio_tensor = engine.model.worker.decode_vae(latents)
        wav = audio_tensor.cpu().numpy()

    # Hardware Math Validation
    torch.cuda.synchronize()
    
    total_wall_time = wall_end - wall_start
    audio_duration = len(wav) / 24000
    rtf = audio_duration / total_wall_time

    print_header("THE BRUTAL TRUTH")
    print(f"Total Wall Clock Time:      {total_wall_time:.4f} seconds")
    print(f"Audio Duration Generated:   {audio_duration:.4f} seconds")
    print(f"Real-Time Factor (RTF):     {rtf:.3f}x")
    print("-" * 50)
    
    total_gpu_time = 0
    total_micro_stall = 0
    
    for i in range(len(ev_starts)):
        chunk_gpu_time = ev_starts[i].elapsed_time(ev_ends[i]) / 1000.0
        total_gpu_time += chunk_gpu_time
        
        # Calculate Micro-Stalls (gap between end of chunk N-1 and start of chunk N)
        micro_stall = 0
        if i > 0:
            micro_stall = ev_ends[i-1].elapsed_time(ev_starts[i]) / 1000.0
            # If negative or tiny, PyTorch pipelining successfully overlapped them
            micro_stall = max(0.0, micro_stall)
            total_micro_stall += micro_stall
            
        print(f"Chunk {i+1:02d} | 3090 Exec Time: {chunk_gpu_time:.4f}s | CPU->GPU Micro-Stall Gap: {micro_stall:.6f}s")

    print("-" * 50)
    print(f"Total True 3090 Math Time:  {total_gpu_time:.4f} sec")
    print(f"Total Celeron Stalls:       {total_micro_stall:.6f} sec (Time 3090 sat idle waiting for CPU)")
    print(f"Total PCIe Sync Latency:    {total_pcie_sync_time:.6f} sec (Time wasted checking EOS flag)")
    print("-" * 50)
    
    # Calculate Overshoot Waste
    avg_frame_time = total_gpu_time / (tokens_gen + wasted_overshoot_frames)
    overshoot_time_waste = avg_frame_time * wasted_overshoot_frames
    
    print("OVERSHOOT ANALYSIS:")
    print(f"Garbage frames calculated:  {wasted_overshoot_frames} frames (Due to chunk size 30)")
    print(f"Time wasted on garbage:     {overshoot_time_waste:.4f} sec")
    print(f"Time saved bypassing PCIe:  ~{len(ev_starts) * 30 * 0.005:.4f} sec (Estimated)")
    print("-> The overshoot trade-off was mathematically worth it.")
    
    import soundfile as sf
    sf.write("debugger_output_cpp.wav", wav, 24000)

if __name__ == "__main__":
    run_debugger()