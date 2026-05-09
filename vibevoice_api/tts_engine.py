# --- START OF FILE vibevoice_api/tts_engine.py ---
from __future__ import annotations
import os
import torch
import numpy as np
import logging
import io
import time
import base64
from typing import Optional, Tuple, List, AsyncIterator

import soundfile as sf
import librosa

from vibevoice_api.config import CONFIG
from vibevoice_api.audio_utils import apply_speed, to_bytes_for_format
from vibevoice_api.voice_map import VoiceMapper
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.tokenizer import MMEmbedding

log = logging.getLogger("vibevoice_api.tts_engine")
_engine_instance = None


def load_audio_norm(path_or_bytes) -> np.ndarray:
    """Loads and strictly normalizes audio to -25dBFS just like upstream VibeVoice"""
    if isinstance(path_or_bytes, bytes):
        wav, sr = sf.read(io.BytesIO(path_or_bytes), dtype="float32")
    else:
        wav, sr = sf.read(path_or_bytes, dtype="float32")
        
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 24000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
        
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


class EngineState:
    def __init__(self, llm_path: str, diff_path: str):
        log.info(f"Loading native ExLlamaV3 VibeVoice model from {llm_path}...")
        self.config = Config.from_directory(llm_path)
        self.model = Model.from_config(self.config)
        self.model.load(diffusion_model_path=diff_path)
        self.tokenizer = Tokenizer.from_config(self.config)
        
        self.speech_start_id = self.tokenizer.single_id("<|vision_start|>")
        self.speech_end_id = self.tokenizer.single_id("<|vision_end|>")


def _get_engine() -> EngineState:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EngineState(CONFIG.llm_model_path, CONFIG.diffusion_model_path)
    return _engine_instance


def _create_cache(model, max_num_tokens):
    cache = Cache(model, max_num_tokens=max_num_tokens)
    for module in model.get_cache_layers():
        cache.layers[module.layer_idx].alloc(module.device)
    return cache


def _destroy_cache(cache, model):
    for module in model.get_cache_layers():
        cache.layers[module.layer_idx].free()
    cache.detach_from_model(model)


def synthesize(
    *,
    root_dir: str, text: str, voice: Optional[str] = "Alice",
    voice_path: Optional[str] = None, voice_data_b64: Optional[str] = None,
    speakers: Optional[List[str]] = None, response_format: str = "wav",
    speed: Optional[float] = None, seed: Optional[int] = None,
    cfg_scale: Optional[float] = None, ddpm_steps: Optional[int] = None,
    use_sampling: Optional[bool] = None, temperature: Optional[float] = None,
    top_p: Optional[float] = None, negative_llm_steps_to_cache: Optional[int] = None,
    increase_cfg: Optional[bool] = None, split_by_newline: Optional[bool] = None,
) -> Tuple[bytes, str]:
    
    engine = _get_engine()
    
    # 1. Resolve Voice Audio
    audio_source = None
    if voice_data_b64:
        if voice_data_b64.startswith("data:"): voice_data_b64 = voice_data_b64.split(",", 1)[1]
        audio_source = base64.b64decode(voice_data_b64)
    elif voice_path:
        audio_source = voice_path
    elif voice:
        mapper = VoiceMapper(root_dir)
        audio_source = mapper.resolve(voice)
        
    if not audio_source: raise ValueError("Could not resolve reference voice.")
        
    # 2. Encode Real Human Voice (Moved to CPU for ExLlamaV3 text-stitching)
    wav_norm = load_audio_norm(audio_source)
    with torch.inference_mode():
        device = engine.model.output_device or "cuda:0"
        audio_tensor = torch.from_numpy(wav_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        voice_embeddings = engine.model.worker.encode_acoustic(audio_tensor).cpu()

    token_string = torch.full((1, voice_embeddings.shape[1]), -1, dtype=torch.long)
    voice_mme = MMEmbedding(embeddings=voice_embeddings.squeeze(0).half(), token_string=token_string, text_alias="<$VOICE$>")
    
    # 3. Format Prompt
    prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
    prompt += " Voice input:\n Speaker 0:<|vision_start|><$VOICE$><|vision_end|>\n"
    prompt += f" Text input:\n Speaker 0: {text.strip()}\n Speech output:\n<|vision_start|>"
    
    input_ids = engine.tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True, embeddings=[voice_mme])
    
    cfg = cfg_scale if cfg_scale is not None else CONFIG.cfg_scale
    use_cfg = cfg > 1.0
    seed = seed if seed is not None and seed != -1 else int(time.time())
    increase_cfg = increase_cfg if increase_cfg is not None else CONFIG.increase_cfg
    
    with torch.inference_mode():
        # 4. Static Negative CFG condition
        if use_cfg:
            neg_input_ids = torch.tensor([[engine.speech_start_id]], dtype=torch.long, device="cpu")
            inputs_embeds_neg = engine.model.modules[0].forward(neg_input_ids, {})
            _, neg_hidden = engine.model.forward(inputs_embeds=inputs_embeds_neg, params={"attn_mode": "flash_attn_nc"})
            cond_neg = neg_hidden[:, -1:, :].half()
        else:
            cond_neg = None
            
        cache_pos = _create_cache(engine.model, max_num_tokens=8192)

        try:
            # 5. LLM Prompt Prefill
            inputs_embeds_pos = engine.model.modules[0].forward(input_ids, {"indexed_embeddings": [voice_mme]})
            params_pos = {"attn_mode": "flash_attn", "cache": cache_pos, "past_len": 0, "batch_shape": (1, 8192)}
            logits_pos, hidden_last_pos = engine.model.forward(inputs_embeds=inputs_embeds_pos, params=params_pos)
            
            past_len = inputs_embeds_pos.shape[1]
            all_latents = []
            log.info("Starting autoregressive diffusion loop (Pipelined)...")
            
            # 6. Pipelined AR Loop (Solves CPU Chit-Chat)
            chunk_size = 30  # Number of frames the CPU will queue onto the GPU at once
            eos_flag = torch.zeros(1, dtype=torch.bool, device=device)
            
            for chunk_start in range(0, 1500, chunk_size):
                chunk_latents = []
                chunk_preds = []
                
                # The Celeron queues this entire loop into CUDA instantly without waiting.
                for t in range(chunk_start, min(chunk_start + chunk_size, 1500)):
                    cond_pos = hidden_last_pos[:, -1:, :].half()
                    
                    # DiT (C++)
                    z = engine.model.worker.sample_latent(cond_pos, cond_neg if use_cfg else cond_pos, cfg, seed + t, increase_cfg)
                    chunk_latents.append(z)
                    
                    # Acoustic Connector (C++)
                    step_embed = engine.model.worker.acoustic_connector_forward(z.squeeze(1)).unsqueeze(1)
                    
                    # LLM Step (C++ Kernels via Python dispatch)
                    params_pos = {"attn_mode": "flash_attn", "cache": cache_pos, "past_len": past_len, "batch_shape": (1, 8192)}
                    logits_pos, hidden_last_pos = engine.model.forward(inputs_embeds=step_embed.to(inputs_embeds_pos.dtype), params=params_pos)
                    past_len += 1
                    
                    # Async EOS Check (NO .item() inside the inner loop!)
                    pred_id = logits_pos[0, -1, :].argmax()
                    chunk_preds.append(pred_id)
                    eos_flag.logical_or_(pred_id == engine.speech_end_id)
                
                all_latents.extend(chunk_latents)
                
                # Single PCIe Sync Point per chunk. 
                # This drops CPU<->GPU communication overhead by ~97%
                if eos_flag.item():
                    # Find exact EOS frame to trim any over-generated garbage latents
                    preds_cpu = torch.stack(chunk_preds).cpu()
                    eos_indices = (preds_cpu == engine.speech_end_id).nonzero(as_tuple=True)[0]
                    if len(eos_indices) > 0:
                        first_eos_idx = eos_indices[0].item()
                        trim_count = len(chunk_preds) - first_eos_idx - 1
                        if trim_count > 0:
                            all_latents = all_latents[:-trim_count]
                    break
        finally:
            _destroy_cache(cache_pos, engine.model)

        if not all_latents: return b"", "audio/wav"

        # 7. VAE Decode (100% C++)
        latents = torch.cat(all_latents, dim=1)
        audio_tensor = engine.model.worker.decode_vae(latents)

    # 8. Post-process Audio
    wav = audio_tensor.cpu().numpy()
    warmup = 2400
    if len(wav) > warmup:
        mask = np.abs(wav[warmup:]) > 0.005
        trim_start = max(warmup, warmup + np.argmax(mask) - 800) if np.any(mask) else warmup
        
        mask_tail = np.abs(wav) > 0.01
        trim_end = min(len(wav), len(wav) - 1 - np.argmax(mask_tail[::-1]) + 1200 + 1) if np.any(mask_tail) else len(wav)
        
        wav = wav[trim_start:trim_end] if trim_start < trim_end else wav[warmup:]
        
        n_in = min(480, len(wav))
        if n_in > 0: wav[:n_in] *= (np.linspace(0, 1, n_in, dtype=np.float32) ** 2)
            
        n_out = min(1200, len(wav))
        if n_out > 0: wav[-n_out:] *= np.linspace(1, 0, n_out, dtype=np.float32)

    if speed is not None:
        wav = apply_speed(wav, float(speed))

    max_val = np.max(np.abs(wav))
    if max_val > 0.95:
        wav = wav / (max_val / 0.95)

    data, content_type = to_bytes_for_format(wav, CONFIG.sample_rate, response_format)
    return data, content_type


async def synthesize_stream_pcm(*args, **kwargs) -> AsyncIterator[np.ndarray]:
    data, _ = synthesize(**kwargs)
    import soundfile as sf
    wav, _ = sf.read(io.BytesIO(data), dtype="float32")
    yield wav