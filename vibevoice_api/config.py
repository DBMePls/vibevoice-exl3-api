# --- START OF FILE vibevoice_api/config.py ---
from __future__ import annotations

import os
from dataclasses import dataclass

# ======================================================================================
# NOTE: The default values below are configured for the "High Quality" preset.
# They can be overridden by setting the corresponding environment variables.
# ======================================================================================

@dataclass(frozen=True)
class ServerConfig:
    host: str = os.environ.get("VIBEVOICE_API_HOST", "0.0.0.0")
    port: int = int(os.environ.get("VIBEVOICE_API_PORT", "8000"))
    base_path: str = os.environ.get("VIBEVOICE_API_BASE_PATH", "/v1")

    # --- MODEL & PERFORMANCE CONFIGURATION ---
    diffusion_model_path: str = os.environ.get("VIBEVOICE_DIFFUSION_MODEL", "tensorbanana/vibevoice-7b-no-llm-bf16")
    llm_model_path: str = os.environ.get("VIBEVOICE_LLM_MODEL", "tensorbanana/vibevoice-7b-exl3-8bit")
    quantization_mode: str = os.environ.get("VIBEVOICE_QUANT_MODE", "bf16")
    attention_type: str = os.environ.get("VIBEVOICE_ATTENTION_TYPE", "flash_attention_2")
    torch_compile_mode: str = os.environ.get("VIBEVOICE_TORCH_COMPILE", "max-autotune")
    device_preference: str = os.environ.get("VIBEVOICE_DEVICE", "auto")
    max_concurrency: int = int(os.environ.get("VIBEVOICE_MAX_CONCURRENCY", "1"))

    # --- GENERATION DEFAULTS (High-Quality Preset) ---
    sample_rate: int = int(os.environ.get("VIBEVOICE_SAMPLE_RATE", "24000"))
    ddpm_steps: int = int(os.environ.get("VIBEVOICE_DDPM_STEPS", "16"))
    cfg_scale: float = float(os.environ.get("VIBEVOICE_CFG_SCALE", "1.3"))
    seed: int = int(os.environ.get("VIBEVOICE_SEED", "42")) # -1 for random
    use_sampling: bool = os.environ.get("VIBEVOICE_USE_SAMPLING", "false").lower() in ("true", "1", "yes")
    temperature: float = float(os.environ.get("VIBEVOICE_TEMPERATURE", "0.95"))
    top_p: float = float(os.environ.get("VIBEVOICE_TOP_P", "0.95"))
    
    # -------------------------------------------------------------------------
    # NEW DEFAULT: 4 (Lazy Mini-Prefill for the negative CFG state)
    # -------------------------------------------------------------------------
    negative_llm_steps_to_cache: int = int(os.environ.get("VIBEVOICE_NEG_CACHE_STEPS", "2"))
    
    increase_cfg: bool = os.environ.get("VIBEVOICE_INCREASE_CFG", "false").lower() in ("true", "1", "yes")
    split_by_newline: bool = os.environ.get("VIBEVOICE_SPLIT_TEXT", "false").lower() in ("true", "1", "yes")
    
    # --- LOGGING & MISC ---
    logs_dir: str = os.environ.get("VIBEVOICE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    log_prompts: bool = os.environ.get("VIBEVOICE_LOG_PROMPTS", "1") not in {"0", "false", "False"}
    prompt_maxlen: int = int(os.environ.get("VIBEVOICE_PROMPT_MAXLEN", "4096"))
    instructions_maxlen: int = int(os.environ.get("VIBEVOICE_INSTRUCTIONS_MAXLEN", "2000"))
    instructions_strategy: str = os.environ.get("VIBEVOICE_INSTRUCTIONS_STRATEGY", "system_only")
    instructions_repeat: int = int(os.environ.get("VIBEVOICE_INSTRUCTIONS_REPEAT", "1"))
    sse_chunk_bytes: int = int(os.environ.get("VIBEVOICE_SSE_CHUNK_BYTES", "16384"))
    
    # --- FFMPEG SETTINGS ---
    ffmpeg_path: str = os.environ.get("VIBEVOICE_FFMPEG", "ffmpeg")
    ffmpeg_bitrate: str = os.environ.get("VIBEVOICE_FFMPEG_BITRATE", "")
    flac_level: str = os.environ.get("VIBEVOICE_FLAC_LEVEL", "")
    opus_container: str = os.environ.get("VIBEVOICE_OPUS_CONTAINER", "webm")
    opus_vbr_mode: str = os.environ.get("VIBEVOICE_OPUS_VBR", "vbr")
    opus_application: str = os.environ.get("VIBEVOICE_OPUS_APPLICATION", "audio")
    opus_frame_duration: str = os.environ.get("VIBEVOICE_OPUS_FRAME_DURATION", "")
    aac_profile: str = os.environ.get("VIBEVOICE_AAC_PROFILE", "")
    aac_mode: str = os.environ.get("VIBEVOICE_AAC_MODE", "cbr")
    aac_q: str = os.environ.get("VIBEVOICE_AAC_Q", "")

    # --- AUTHENTICATION ---
    require_api_key: bool = os.environ.get("VIBEVOICE_REQUIRE_API_KEY", "0") not in {"0", "false", "False"}
    admin_token: str = os.environ.get("VIBEVOICE_ADMIN_TOKEN", "")
    keystore_path: str = os.environ.get("VIBEVOICE_KEYSTORE", os.path.join(os.getcwd(), "logs", "keys.json"))

CONFIG = ServerConfig()
# --- END OF FILE vibevoice_api/config.py ---