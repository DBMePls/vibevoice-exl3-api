# --- START OF FILE hardware_profiler.py ---
import os
import sys
import time
import torch
import numpy as np
import subprocess
import logging
import traceback
import gc

print("\n>>> BOOTING HARDWARE PROFILER (MULTI-CHUNK TESTER)...", flush=True)

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
    print(f"\n{'='*95}")
    print(f"{title:^95}")
    print(f"{'='*95}", flush=True)

def load_audio_ffmpeg_strict(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ar", "24000", "-ac", "1", "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    wav = wav * ((10 ** (-25.0 / 20.0)) / (np.sqrt(np.mean(wav**2)) + 1e-6))
    maxabs = np.max(np.abs(wav))
    if maxabs > 1.0: wav /= (maxabs + 1e-6)
    return wav

def run_debugger():
    try:
        chunk_sizes_to_test = [1, 2, 4, 8, 16, 32, 64]
        cfg = 1.3
        use_cfg = True
        seed = 999 
        max_frames = 1500
        cache_len = 8192
        
        print_header("1. ENVIRONMENT & SETTINGS")
        print(f"DiT Model Path:    {os.environ.get('VIBEVOICE_DIFFUSION_MODEL', 'Not Set')}")
        print(f"LLM Model Path:    {os.environ.get('VIBEVOICE_LLM_MODEL', 'Not Set')}")
        print(f"Chunk Sizes:       {chunk_sizes_to_test}")
        print(f"Fixed Seed:        {seed} (Ensures 1:1 comparison)", flush=True)

        t0 = time.perf_counter()
        engine = tts_engine._get_engine()
        device = engine.model.output_device or "cuda:0"
        print(f"Hardware Device:   {device}")
        print(f"[TIME] Engine loaded in {(time.perf_counter()-t0):.2f} seconds.", flush=True)

        text = (
            "This is a comprehensive stress test of the Vibe Voice generation system. "
            "We are using a much longer prompt containing multiple sentences to ensure that the hardware is fully saturated. "
            "By generating a significant amount of audio, we can accurately measure the true impact of PCIe synchronization latency "
            "versus the penalty of overshooting."
        )
        
        voice_path = os.path.join(os.getcwd(), "demo", "voices", "en-Alice_woman.wav")
        if not os.path.exists(voice_path):
            print(f"\n[CRITICAL ERROR] Cannot find audio file at {voice_path}!")
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
        
        with torch.inference_mode():
            if use_cfg:
                neg_input_ids = torch.tensor([[engine.speech_start_id]], dtype=torch.long, device="cpu")
                inputs_embeds_neg = engine.model.modules[0].forward(neg_input_ids, {})
                _, neg_hidden = engine.model.forward(inputs_embeds=inputs_embeds_neg, params={"attn_mode": "flash_attn_nc"})
                cond_neg = neg_hidden[:, -1:, :].half()
            else:
                cond_neg = None

        summary_metrics = []
        final_wav = None

        for chunk_size in chunk_sizes_to_test:
            print_header(f"TESTING CHUNK SIZE: {chunk_size}")
            
            with torch.inference_mode():
                cache_pos = _create_cache(engine.model, max_num_tokens=cache_len)

                try:
                    inputs_embeds_pos = engine.model.modules[0].forward(input_ids, {"indexed_embeddings": [voice_mme]})
                    params_pos = {"attn_mode": "flash_attn", "cache": cache_pos, "past_len": 0, "batch_shape": (1, cache_len)}
                    logits_pos, hidden_last_pos = engine.model.forward(inputs_embeds=inputs_embeds_pos, params=params_pos)
                    
                    past_len = inputs_embeds_pos.shape[1]
                    all_latents = []
                    eos_flag = torch.zeros(1, dtype=torch.bool, device=device)
                    tokens_gen = 0
                    
                    total_pcie_sync_time = 0.0
                    total_cpu_queue_time = 0.0
                    wasted_overshoot_frames = 0
                    ev_starts = []
                    ev_ends = []
                    
                    wall_start = time.perf_counter()
                    
                    sys.stdout.write("Generating: [")
                    sys.stdout.flush()
                    
                    for chunk_start in range(0, max_frames, chunk_size):
                        chunk_latents = []
                        chunk_preds = []
                        
                        t_cpu_start = time.perf_counter()
                        
                        ev_start = torch.cuda.Event(enable_timing=True)
                        ev_end = torch.cuda.Event(enable_timing=True)
                        ev_start.record()
                        
                        for t in range(chunk_start, min(chunk_start + chunk_size, max_frames)):
                            cond_pos = hidden_last_pos[:, -1:, :].half()
                            
                            z = engine.model.worker.sample_latent(cond_pos, cond_neg if use_cfg else cond_pos, cfg, seed + t, False)
                            chunk_latents.append(z)
                            step_embed = engine.model.worker.acoustic_connector_forward(z.squeeze(1)).unsqueeze(1)
                            
                            params_pos = {"attn_mode": "flash_attn", "cache": cache_pos, "past_len": past_len, "batch_shape": (1, cache_len)}
                            logits_pos, hidden_last_pos = engine.model.forward(inputs_embeds=step_embed.to(inputs_embeds_pos.dtype), params=params_pos)
                            
                            past_len += 1
                            tokens_gen += 1
                            
                            pred_id = logits_pos[0, -1, :].argmax()
                            chunk_preds.append(pred_id)
                            eos_flag.logical_or_(pred_id == engine.speech_end_id)
                        
                        ev_end.record()
                        ev_starts.append(ev_start)
                        ev_ends.append(ev_end)
                        
                        t_cpu_end = time.perf_counter()
                        total_cpu_queue_time += (t_cpu_end - t_cpu_start)
                        
                        if chunk_size >= 8 or (chunk_start % 8 == 0):
                            sys.stdout.write("#")
                            sys.stdout.flush()
                        
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
                    sys.stdout.write("] Done!\n")
                    sys.stdout.flush()
                    
                finally:
                    _destroy_cache(cache_pos, engine.model)

                latents = torch.cat(all_latents, dim=1)
                audio_tensor = engine.model.worker.decode_vae(latents)
                final_wav = audio_tensor.cpu().numpy()

            torch.cuda.synchronize()
            
            total_wall_time = wall_end - wall_start
            audio_duration = len(final_wav) / 24000
            rtf = audio_duration / total_wall_time
            total_gpu_time = sum(s.elapsed_time(e) for s, e in zip(ev_starts, ev_ends)) / 1000.0
            
            total_micro_stall = 0.0
            for i in range(1, len(ev_starts)):
                micro_stall = max(0.0, ev_ends[i-1].elapsed_time(ev_starts[i]) / 1000.0)
                total_micro_stall += micro_stall
                
            avg_frame_time = total_gpu_time / (tokens_gen + wasted_overshoot_frames)
            overshoot_time_waste = avg_frame_time * wasted_overshoot_frames

            summary_metrics.append({
                "chunk": chunk_size,
                "rtf": rtf,
                "wall_time": total_wall_time,
                "gpu_time": total_gpu_time,
                "pcie_sync": total_pcie_sync_time,
                "cpu_stalls": total_micro_stall,
                "wasted_frames": wasted_overshoot_frames,
                "wasted_time": overshoot_time_waste
            })
            
            print(f"Wall Time: {total_wall_time:.3f}s | GPU Math: {total_gpu_time:.3f}s | RTF: {rtf:.3f}x")

            del all_latents, latents, audio_tensor, cache_pos, ev_starts, ev_ends
            gc.collect()
            torch.cuda.empty_cache()

        print_header("FINAL HEAVY-DUTY CHUNK SIZE COMPARISON TABLE")
        print(f"{'Chunk':<7} | {'RTF':<7} | {'Wall Time':<11} | {'GPU Math Time':<15} | {'PCIe Sync':<11} | {'Micro-Stalls':<14} | {'Garbage Waste':<15}")
        print("-" * 95)
        
        best_rtf = 0
        best_chunk = 1
        
        for m in summary_metrics:
            if m["rtf"] > best_rtf:
                best_rtf = m["rtf"]
                best_chunk = m["chunk"]
                
            print(f"{m['chunk']:<7} | {m['rtf']:<6.3f}x | {m['wall_time']:<8.3f} s | {m['gpu_time']:<10.3f} s | {m['pcie_sync']:<6.4f} s | {m['cpu_stalls']:<9.5f} s | {m['wasted_frames']:<2} fr ({m['wasted_time']:<5.3f}s)")

        print("-" * 95)
        print(f"\n[WINNER] The optimal chunk_size for your hardware on LONG generations is: {best_chunk} (RTF: {best_rtf:.3f}x)")
        print(f"-> You should set `chunk_size = {best_chunk}` in your API server.")

    except Exception as e:
        print("\n[CRITICAL FAILURE] The profiler crashed with the following error:")
        traceback.print_exc()

if __name__ == "__main__":
    run_debugger()