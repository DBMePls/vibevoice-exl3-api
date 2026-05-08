from __future__ import annotations
import threading
import sys
import os
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, AsyncIterator
import numpy as np
import torch
import torch.nn.functional as F
import logging

# Add vvembed to Python Path
VIBEVOICE_API_ROOT = os.path.dirname(os.path.abspath(__file__))
VVEMBED_PATH = os.path.join(VIBEVOICE_API_ROOT, '..', 'vvembed')
if VVEMBED_PATH not in sys.path:
    sys.path.insert(0, VVEMBED_PATH)

from vibevoice_api.audio_utils import apply_speed, to_bytes_for_format
from vibevoice_api.config import CONFIG
from vibevoice_api.voice_map import VoiceMapper
from vibevoice_api import observability as obs

from vvembed.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference, ExLlamaV3Wrapper
from vvembed.processor.vibevoice_processor import VibeVoiceProcessor

log = logging.getLogger("vibevoice_api.tts_engine")

@dataclass
class LoadedModel:
    processor: VibeVoiceProcessor
    model: VibeVoiceForConditionalGenerationInference
    device: str
    torch_dtype: torch.dtype
    sample_rate: int
    semaphore: threading.Semaphore

_engine_lock = threading.Lock()
_model_cache: Dict[str, LoadedModel] = {}

def _select_device() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        log.info("CUDA device found. Using 'cuda' with bfloat16.")
        return "cuda", torch.bfloat16
    log.warning("CUDA not found. Running on CPU. This will be very slow.")
    return "cpu", torch.float32

def _load_exllama_model(llm_model_path: str) -> ExLlamaV3Wrapper:
    try:
        from exllamav3 import Config, Model, Cache
    except ImportError as e:
        log.error("Fatal: exllamav3 is not installed.")
        raise e

    log.info(f"Loading exllamav3 model from: {llm_model_path}")
    
    if not os.path.isdir(llm_model_path):
        from huggingface_hub import snapshot_download
        llm_model_path = snapshot_download(repo_id=llm_model_path, local_dir_use_symlinks=False)

    if os.path.exists(os.path.join(llm_model_path, "snapshots")):
        snapshot_dirs = os.listdir(os.path.join(llm_model_path, "snapshots"))
        if snapshot_dirs:
            llm_model_path = os.path.join(llm_model_path, "snapshots", snapshot_dirs[0])
    
    exllama_config = Config.from_directory(llm_model_path)
    exllama_model = Model.from_config(exllama_config)
    exllama_positive_cache = Cache(exllama_model, max_num_tokens=4096)
    exllama_negative_cache = Cache(exllama_model, max_num_tokens=4096)
    exllama_model.load()

    return ExLlamaV3Wrapper(
        model=exllama_model,
        positive_cache=exllama_positive_cache,
        negative_cache=exllama_negative_cache,
        config=exllama_config
    )

def _load_model(diffusion_model_path: str, llm_model_path: str, quant_mode: str, attention_type: str) -> LoadedModel:
    cache_key = f"{diffusion_model_path}|{llm_model_path}|{quant_mode}|{attention_type}"
    with _engine_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

        device, torch_dtype = _select_device()
        model_kwargs = {"device_map": {"": device}, "torch_dtype": torch.bfloat16}

        # 1. Load LLM via ExLlama
        try:
            exllama_wrapper = _load_exllama_model(llm_model_path)
        except Exception as e:
            log.error(f"Failed to load exllamav3: {e}")
            raise RuntimeError(f"Could not load exllamav3 from {llm_model_path}") from e
        
        # 2. Load Diffusion Model
        log.info(f"Loading Diffusion Model from '{diffusion_model_path}'")
        processor = VibeVoiceProcessor.from_pretrained(diffusion_model_path)
        
        model_kwargs["attn_implementation"] = "sdpa"
        
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            diffusion_model_path,
            ignore_mismatched_sizes=True,
            **model_kwargs,
        )

        model.exllama = exllama_wrapper
        model.eval()
        model.set_ddpm_inference_steps(num_steps=CONFIG.ddpm_steps)

        loaded = LoadedModel(
            processor=processor,
            model=model,
            device=device,
            torch_dtype=torch_dtype,
            sample_rate=CONFIG.sample_rate,
            semaphore=threading.Semaphore(max(1, int(CONFIG.max_concurrency))),
        )
        _model_cache[cache_key] = loaded
        return loaded

def synthesize(
    *,
    root_dir: str,
    text: str,
    voice: Optional[str],
    voice_path: Optional[str] = None,
    voice_data_b64: Optional[str] = None,
    speakers: Optional[List[str]] = None,
    response_format: str = "wav",
    speed: Optional[float] = None,
    seed: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    ddpm_steps: Optional[int] = None,
    use_sampling: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    negative_llm_steps_to_cache: Optional[int] = None,
    increase_cfg: Optional[bool] = None,
    split_by_newline: Optional[bool] = None,
) -> Tuple[bytes, str]:
    state = _load_model(CONFIG.diffusion_model_path, CONFIG.llm_model_path, CONFIG.quantization_mode, CONFIG.attention_type)

    mapper = VoiceMapper(root_dir)
    voice_samples = None
    if voice_path:
        voice_samples = [voice_path]
    elif voice:
        resolved_path = mapper.resolve(voice)
        if resolved_path:
            voice_samples = [resolved_path]

    current_seed = seed if seed is not None else CONFIG.seed
    if current_seed == -1:
        current_seed = np.random.randint(0, 2**32 - 1)
    torch.manual_seed(current_seed)

    text_chunks = [text.strip()]

    all_audio_segments = []

    with state.semaphore, torch.inference_mode():
        state.model.set_ddpm_inference_steps(ddpm_steps or CONFIG.ddpm_steps)
        
        gen_config = {}
        if use_sampling or CONFIG.use_sampling:
            gen_config['do_sample'] = True
            gen_config['temperature'] = temperature or CONFIG.temperature
            gen_config['top_p'] = top_p or CONFIG.top_p
        else:
            gen_config['do_sample'] = False

        for i, chunk_text in enumerate(text_chunks):
            formatted_text = f"Speaker 1: {chunk_text}\n"
            
            inputs = state.processor(
                text=[formatted_text],
                voice_samples=[voice_samples] if voice_samples else [None],
                return_tensors="pt",
            ).to(state.model.device)

            with obs.observe_latency(obs.SYNTHESIS_LATENCY, (response_format or "wav").lower()):
                obs.ACTIVE_INFERENCES.inc()
                try:
                    outputs = state.model.generate(
                        **inputs,
                        cfg_scale=cfg_scale or CONFIG.cfg_scale,
                        seed=current_seed,
                        generation_config=gen_config,
                        tokenizer=state.processor.tokenizer,
                        verbose=False,
                    )
                finally:
                    obs.ACTIVE_INFERENCES.dec()
            
            speech_tensor = outputs.speech_outputs[0]
            if speech_tensor is not None and speech_tensor.numel() > 0:
                all_audio_segments.append(speech_tensor.cpu())

    if not all_audio_segments:
        wav = np.zeros(int(0.5 * state.sample_rate), dtype=np.float32)
    else:
        final_audio_tensor = torch.cat(all_audio_segments, dim=-1)
        wav = final_audio_tensor.to(torch.float32).squeeze().numpy()

    if speed is not None:
        wav = apply_speed(wav, float(speed))

    data, content_type = to_bytes_for_format(wav, state.sample_rate, response_format)
    return data, content_type

async def synthesize_stream_pcm(*args, **kwargs) -> AsyncIterator[np.ndarray]:
    data, _ = synthesize(**kwargs)
    import soundfile as sf
    import io
    wav, _ = sf.read(io.BytesIO(data), dtype="float32")
    yield wav