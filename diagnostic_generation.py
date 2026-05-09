import os
import sys
import time
import torch
import numpy as np
import subprocess

# Ensure we are in the root directory
sys.path.insert(0, os.getcwd())

from vibevoice_api import tts_engine
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

def print_mem(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"[MEMORY] {label:<30} | Allocated: {allocated:7.2f} MB | Reserved: {reserved:7.2f} MB")

def print_tensor_stats(name, t):
    if t is None:
        print(f"[TENSOR] {name}: None")
        return
    
    t_f = t.float() if t.is_floating_point() else t.float()
    nans = torch.isnan(t_f).sum().item()
    infs = torch.isinf(t_f).sum().item()
    rms = torch.sqrt(torch.mean(t_f**2)).item()
    
    print(f"[TENSOR] {name}")
    print(f"         |- Shape : {list(t.shape)}")
    print(f"         |- Dtype : {t.dtype} | Device: {t.device}")
    print(f"         |- Min   : {t_f.min().item():10.5f}  | Max : {t_f.max().item():10.5f}")
    print(f"         |- Mean  : {t_f.mean().item():10.5f}  | Std : {t_f.std().item():10.5f} | RMS : {rms:10.5f}")
    
    if nans > 0 or infs > 0:
        print(f"         |- WARNING: {nans} NaNs, {infs} Infs detected!")

# --- BULLETPROOF AUDIO LOADER ---

def load_audio_ffmpeg_strict(path: str) -> np.ndarray:
    """Uses FFMPEG to force 24kHz, Mono, Float32 PCM extraction."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", path,
        "-ar", "24000", "-ac", "1", "-f", "f32le", "-"
    ]
    print(f"[FFMPEG] Executing: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    
    # Strict -25 dBFS Normalization
    target_db_fs = -25.0
    eps = 1e-6
    rms = np.sqrt(np.mean(wav**2))
    target_lin = 10 ** (target_db_fs / 20.0)
    scalar = target_lin / (rms + eps)
    wav = wav * scalar
    
    maxabs = np.max(np.abs(wav))
    if maxabs > 1.0:
        wav /= (maxabs + eps)
        
    return wav

# --- MAIN DEBUGGER ---

def run_debugger():
    print_header("VIBEVOICE FORENSIC DIAGNOSTIC PIPELINE")
    print_mem("Baseline")

    print_header("1. ENGINE INITIALIZATION")
    t0 = time.perf_counter()
    engine = tts_engine._get_engine()
    t1 = time.perf_counter()
    print(f"[TIME] Engine loaded in {(t1-t0):.2f} seconds.")

    text = "This is a diagnostic test of the VibeVoice system. We are tracking all internal metrics to ensure pristine audio generation."
    
    print_header("2. FFMPEG AUDIO EXTRACTION & ENCODING")
    voice_path = os.path.join(os.getcwd(), "demo", "voices", "en-Alice_woman.wav")
    print(f"[INFO] Reference voice: {voice_path}")
    
    if not os.path.exists(voice_path):
        print(f"[ERROR] Could not find voice file at {voice_path}. Exiting.")
        return
        
    wav_norm = load_audio_ffmpeg_strict(voice_path)
    print(f"[INFO] Strict Audio length: {len(wav_norm)} samples ({(len(wav_norm)/24000):.2f} seconds)")
    
    with torch.inference_mode():
        device = engine.model.output_device or "cuda:0"
        audio_tensor = torch.from_numpy(wav_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Use C++ worker directly!
        ac_emb = engine.model.worker.encode_acoustic(audio_tensor)
        print_tensor_stats("Acoustic Embeddings (C++ Connector)", ac_emb)
        
        voice_embeddings = ac_emb.cpu()
    
    print_tensor_stats("Final Combined Voice Embeddings", voice_embeddings)
    
    token_string = torch.full((1, voice_embeddings.shape[1]), -1, dtype=torch.long)
    voice_mme = MMEmbedding(embeddings=voice_embeddings.squeeze(0).half(), token_string=token_string, text_alias="<$VOICE$>")

    print_header("3. EXACT TOKENIZATION DUMP")
    prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
    prompt += " Voice input:\n Speaker 0:<|vision_start|><$VOICE$><|vision_end|>\n"
    prompt += f" Text input:\n Speaker 0: {text.strip()}\n" 
    prompt += " Speech output:\n<|vision_start|>"
    
    input_ids = engine.tokenizer.encode(
        prompt, 
        add_bos=False, 
        encode_special_tokens=True, 
        embeddings=[voice_mme]
    )
    
    print(f"[INFO] Input prompt tokens length: {input_ids.shape[1]}")
    print("[INFO] BPE Token Breakdown (Look closely at spaces near 'Speaker 0:' and '<|vision_start|>'):")
    
    id_list = input_ids[0].tolist()
    for i, tid in enumerate(id_list):
        if tid == -1 or tid >= 1000000000:
            print(f"  [{i:3d}] ID: {tid:<9d} | String: <$VOICE_EMBEDDING_FRAME$>")
        else:
            string_val = engine.tokenizer.tokenizer.decode([tid])
            if string_val == "":
                string_val = engine.tokenizer.get_id_to_piece_list(True)[tid]
            print(f"  [{i:3d}] ID: {tid:<9d} | String: {repr(string_val)}")

    print_header("4. LLM + DiT AUTOREGRESSIVE STREAM")
    
    cfg_scale = 1.3
    use_cfg = True
    seed = int(time.perf_counter())
    increase_cfg = False
    
    with torch.inference_mode():
        
        # Prepare Static Negative Condition
        if use_cfg:
            bos_id = engine.speech_start_id
            neg_input_ids = torch.tensor([[bos_id]], dtype=torch.long, device="cpu")
            neg_embeds = engine.model.modules[0].forward(neg_input_ids, {})
            params_neg_prefill = {"attn_mode": "flash_attn_nc"}
            _, neg_hidden = engine.model.forward(inputs_embeds=neg_embeds, params=params_neg_prefill)
            cond_neg = neg_hidden[:, -1:, :].half()
            
            print_tensor_stats("Negative CFG Condition (Static)", cond_neg)
        else:
            cond_neg = None

        cache_pos = _create_cache(engine.model, max_num_tokens=8192)

        inputs_embeds_pos = engine.model.modules[0].forward(input_ids, {"indexed_embeddings": [voice_mme]})
        
        print("\n[INFO] Running initial prefill...")
        params_pos = {"attn_mode": "flash_attn", "cache": cache_pos, "past_len": 0, "batch_shape": (1, 8192)}
        logits_pos, hidden_last_pos = engine.model.forward(inputs_embeds=inputs_embeds_pos, params=params_pos)
            
        past_len = inputs_embeds_pos.shape[1]
        all_latents = []
        
        max_speech_frames = 1500
        tokens_gen = 0
        
        print("[INFO] Starting AR loop (Tracking first 5 frames closely):\n")
        
        t_start = time.perf_counter()
        
        for t in range(max_speech_frames):
            cond_pos = hidden_last_pos[:, -1:, :].half()
            c_neg = cond_neg if use_cfg else cond_pos
            
            if t < 5:
                cos_sim = torch.nn.functional.cosine_similarity(cond_pos.float(), c_neg.float(), dim=-1).item()
                print(f"  [Frame {t}] CFG Condition Cosine Similarity (Pos vs Neg): {cos_sim:.5f} (1.0 = identical)")
                
                # Check LLM distribution
                probs = torch.nn.functional.softmax(logits_pos[0, -1, :].float(), dim=-1)
                top_probs, top_ids = torch.topk(probs, 3)
                print(f"  [Frame {t}] LLM Top-3 Tokens:")
                for i in range(3):
                    tid = top_ids[i].item()
                    prob = top_probs[i].item()
                    token_str = engine.tokenizer.get_id_to_piece_list(True)[tid] if tid < 200000 else f"<SPECIAL_{tid}>"
                    print(f"      {i+1}: {token_str!r} ({prob:.2%})")
            
            # C++ DiT
            z = engine.model.worker.sample_latent(cond_pos, c_neg, cfg_scale, seed + t, increase_cfg)
            all_latents.append(z)
            
            # C++ Connector
            step_embed = engine.model.worker.acoustic_connector_forward(z.squeeze(1)).unsqueeze(1)
            
            params_pos = {"attn_mode": "flash_attn", "cache": cache_pos, "past_len": past_len, "batch_shape": (1, 8192)}
            logits_pos, hidden_last_pos = engine.model.forward(inputs_embeds=step_embed.to(inputs_embeds_pos.dtype), params=params_pos)
                
            past_len += 1
            tokens_gen += 1
            
            if tokens_gen > 5 and tokens_gen % 10 == 0:
                print(".", end="", flush=True)
            
            pred_id = logits_pos[0, -1, :].argmax().item()
            if pred_id == engine.speech_end_id:
                print(f"\n[INFO] EOS reached at frame {tokens_gen}")
                break
                
        _destroy_cache(cache_pos, engine.model)

    t_end = time.perf_counter()
    tps = tokens_gen / (t_end - t_start + 1e-6)
    print(f"\n[TIME] Total AR Generation Time:   {t_end-t_start:.2f} s")
    print(f"[METRIC] Frames Generated:         {tokens_gen}")
    print(f"[METRIC] Frames Per Second (FPS):  {tps:.2f} f/s")

    print_header("5. FULL-SEQUENCE C++ VAE DECODER EXECUTION")
    latents = torch.cat(all_latents, dim=1)
    print_tensor_stats("latents", latents)
    
    audio_tensor = engine.model.worker.decode_vae(latents)
    print_tensor_stats("audio_tensor (PCM Float32)", audio_tensor)

    print_header("6. AUDIO CLEANUP & SAVE")
    wav = audio_tensor.cpu().numpy()
    
    # --- Microsoft Reference Audio Cleanup ---
    warmup = 2400
    if len(wav) > warmup:
        mask = np.abs(wav[warmup:]) > 0.005
        if np.any(mask):
            speech_start = np.argmax(mask)
            trim_start = max(warmup, warmup + speech_start - 800)
        else:
            trim_start = warmup
            
        mask_tail = np.abs(wav) > 0.01
        if np.any(mask_tail):
            speech_end = len(wav) - 1 - np.argmax(mask_tail[::-1])
            trim_end = min(len(wav), speech_end + 1200 + 1)
        else:
            trim_end = len(wav)
            
        if trim_start < trim_end:
            wav = wav[trim_start:trim_end]
        else:
            wav = wav[warmup:]
            
        n_in = min(480, len(wav))
        if n_in > 0:
            t = np.linspace(0, 1, n_in, dtype=np.float32)
            wav[:n_in] *= (t * t)
            
        n_out = min(1200, len(wav))
        if n_out > 0:
            t = np.linspace(1, 0, n_out, dtype=np.float32)
            wav[-n_out:] *= t

    # --- Peak Normalization ---
    max_val = np.max(np.abs(wav))
    if max_val > 0.95:
        wav = wav / (max_val / 0.95)

    out_file = "debugger_output_cpp.wav"
    import soundfile as sf
    sf.write(out_file, wav, 24000)
    
    print(f"[SUCCESS] Audio saved to {out_file}")
    print(f"[METRIC] Final Audio Duration: {(len(wav)/24000):.2f} seconds")
    print_header("DIAGNOSTICS COMPLETE")

if __name__ == "__main__":
    run_debugger()