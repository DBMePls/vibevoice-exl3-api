# vibevoice_api/server.py

from __future__ import annotations

import argparse
import logging
import os
import secrets
from typing import Optional

import uuid
import time
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

# Load .env file before importing config
def _load_dotenv_if_present() -> None:
    def _load(path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    import re as _re
                    m = _re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", s)
                    if not m:
                        continue
                    key, val = m.group(1), m.group(2)
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ.setdefault(key, val)
        except Exception:
            pass
    _load(os.path.abspath('.env'))
    _load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))

_load_dotenv_if_present()

import vibevoice_api.config as config_mod
from vibevoice_api.config import CONFIG, ServerConfig
from vibevoice_api.tts_engine import synthesize, synthesize_stream_pcm
from vibevoice_api import auth, observability as obs
from vibevoice_api.voice_map import VoiceMapper

# --- Utility functions for pathing, logging, etc. ---

def _normalize_base_path(raw: str | None) -> str:
    if not raw: return ""
    path = raw.strip()
    if not path or path == "/": return ""
    if not path.startswith("/"): path = "/" + path
    while len(path) > 1 and path.endswith("/"): path = path[:-1]
    return path

API_PREFIX = _normalize_base_path(CONFIG.base_path)

def _join_with_base(path: str) -> str:
    if not path: return API_PREFIX or "/"
    if not path.startswith("/"): path = "/" + path
    if not API_PREFIX: return path
    if path == "/": return API_PREFIX
    return f"{API_PREFIX}{path}"

def _normalize_request_path(path: str) -> str:
    if not path: return "/"
    if not path.startswith("/"): path = "/" + path
    if len(path) > 1 and path.endswith("/"): path = path.rstrip("/")
    return path or "/"

_OPEN_PATHS_RAW = {"/", "/health", "/metrics", "/favicon.ico"}
_OPEN_PATHS = {_normalize_request_path(p) for p in _OPEN_PATHS_RAW}
_OPEN_PATHS.update(_normalize_request_path(_join_with_base(p)) for p in _OPEN_PATHS_RAW)
_ADMIN_KEYS_PREFIX = _normalize_request_path(_join_with_base("/admin/keys"))

def _is_admin_path(path: str) -> bool:
    if not _ADMIN_KEYS_PREFIX or _ADMIN_KEYS_PREFIX == "/": return False
    if path == _ADMIN_KEYS_PREFIX: return True
    return path.startswith(f"{_ADMIN_KEYS_PREFIX}/")

log = logging.getLogger("vibevoice_api")

class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rid = obs.get_request_id()
        except Exception:
            rid = "-"
        record.request_id = rid
        return True

def _configure_logging() -> None:
    if getattr(_configure_logging, "_done", False): return
    handler = logging.StreamHandler()
    handler.addFilter(RequestIDFilter())
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    
    # Get the top-level logger for your application
    app_logger = logging.getLogger("vibevoice_api")
    app_logger.setLevel(logging.INFO)
    app_logger.handlers = [handler]
    
    _configure_logging._done = True

_configure_logging()

router = APIRouter(prefix=API_PREFIX or "")
admin_router = APIRouter(prefix=_ADMIN_KEYS_PREFIX)
app = FastAPI(title="VibeVoice OpenAI-Compatible Audio API (exl3 Backend)")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

log.info(f"Startup config: base_path={API_PREFIX or '/'}, require_api_key={CONFIG.require_api_key}")

# --- UPDATED SpeechRequest Model for Performance Tuning ---

class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to generate audio for")
    model: Optional[str] = Field(None, description="[DEPRECATED] Model is now set on server startup.")
    voice: str = Field("Alice", description="Voice name to use (from voice_map.yaml or demo/voices)")
    voice_path: Optional[str] = Field(None, description="Path to a reference voice sample.")
    voice_data: Optional[str] = Field(None, description="Base64 string or data URL of a reference voice sample.")
    response_format: str = Field("wav", description="Audio format: wav, pcm, mp3, opus, aac")
    speed: Optional[float] = Field(None, description="Playback speed multiplier (0.25 to 4.0)")
    stream_format: Optional[str] = Field(None, description="Set to 'sse' for Server-Sent Events streaming.")
    
    # --- ADVANCED HYPERPARAMETERS ---
    seed: Optional[int] = Field(None, description="Random seed for reproducibility. -1 for random.")
    cfg_scale: Optional[float] = Field(None, description="Classifier-free guidance scale.")
    ddpm_steps: Optional[int] = Field(None, description="Number of diffusion inference steps.")
    use_sampling: Optional[bool] = Field(None, description="Enable stochastic sampling.")
    temperature: Optional[float] = Field(None, description="Sampling temperature (only used if use_sampling is true).")
    top_p: Optional[float] = Field(None, description="Nucleus sampling P value (only used if use_sampling is true).")
    negative_llm_steps_to_cache: Optional[int] = Field(None, description="Cache steps for negative LLM pass. 0=best quality, higher=faster speed.")
    increase_cfg: Optional[bool] = Field(None, description="Experimental: Boost CFG for first 50% of steps for more emotion.")
    split_by_newline: Optional[bool] = Field(None, description="Split text by newlines to process as separate chunks for stability.")

    # Multi-speaker (not fully implemented in this guide's tts_engine for simplicity)
    speakers: Optional[list[str]] = Field(None, description="List of voices for multi-speaker generation.")

# --- Standard API Endpoints ---

@router.get("/")
def root() -> JSONResponse:
    return JSONResponse({"name": "vibevoice_api", "version": "0.2.0-exl3-optimized"})

@router.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok")

@router.get("/metrics")
def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# --- Admin Routes (Unchanged) ---

def _require_admin_auth(request: Request) -> Optional[JSONResponse]:
    token = (CONFIG.admin_token or "").strip()
    if not token:
        return JSONResponse(status_code=403, content={"error": {"message": "Admin token not configured", "type": "admin_disabled"}})
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": {"message": "Admin token required", "type": "invalid_admin_token"}}, headers={"WWW-Authenticate": "Bearer"})
    provided = auth_header.split(" ", 1)[1].strip()
    if not provided or not secrets.compare_digest(provided, token):
        return JSONResponse(status_code=403, content={"error": {"message": "Invalid admin token", "type": "invalid_admin_token"}})
    return None

class AdminKeyCreateRequest(BaseModel):
    key: Optional[str] = Field(None, description="Existing API key to persist. If omitted, a new key is generated.")
    prefix: Optional[str] = Field("sk-", description="Prefix used when generating a new API key if none is supplied.")

@admin_router.get("")
def admin_list_keys(request: Request) -> JSONResponse:
    if (auth_error := _require_admin_auth(request)): return auth_error
    hashes = auth.list_api_key_hashes()
    return JSONResponse({"keys": hashes, "count": len(hashes)})

@admin_router.post("", status_code=201)
def admin_create_key(request: Request, payload: Optional[AdminKeyCreateRequest] = None) -> JSONResponse:
    if (auth_error := _require_admin_auth(request)): return auth_error
    body = payload or AdminKeyCreateRequest()
    key = (body.key or "").strip()
    if not key: key = auth.generate_api_key(prefix=body.prefix if body and body.prefix is not None else "sk-")
    auth.add_api_key(key)
    key_hash = auth.hash_api_key(key)
    log.info("Admin created API key hash=%s", key_hash)
    return JSONResponse({"key": key, "hash": key_hash}, status_code=201)

@admin_router.delete("/{key_hash}")
def admin_delete_key(key_hash: str, request: Request) -> JSONResponse:
    if (auth_error := _require_admin_auth(request)): return auth_error
    normalized = (key_hash or "").strip().lower()
    if not normalized: return JSONResponse(status_code=400, content={"error": {"message": "key_hash is required"}})
    if not auth.remove_api_key(normalized, hashed=True):
        return JSONResponse(status_code=404, content={"error": {"message": "API key not found"}})
    log.info("Admin revoked API key hash=%s", normalized)
    return JSONResponse({"deleted": True, "hash": normalized})

# --- Middleware (Unchanged) ---
@app.middleware("http")
async def metrics_and_request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    obs.set_request_id(rid)
    obs.set_hints_container()
    start = time.perf_counter()
    endpoint = request.url.path
    method = request.method
    normalized_path = _normalize_request_path(endpoint)
    if CONFIG.require_api_key and normalized_path not in _OPEN_PATHS and not _is_admin_path(normalized_path):
        authz = request.headers.get("Authorization", "")
        if not authz.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": {"message": "Unauthorized"}}, headers={"WWW-Authenticate": "Bearer"})
        api_key = authz.split(" ", 1)[1].strip()
        if not auth.validate_api_key(api_key):
            return JSONResponse(status_code=401, content={"error": {"message": "Invalid API key"}})
    try:
        response = await call_next(request)
    except Exception as e:
        obs.ERRORS_TOTAL.labels(type=type(e).__name__).inc()
        log.exception("unhandled exception")
        raise
    finally:
        obs.REQUEST_LATENCY.labels(endpoint, method).observe(time.perf_counter() - start)
    response.headers["X-Request-ID"] = rid
    hints = obs.get_hints()
    if hints:
        response.headers["X-Hints"] = " | ".join(hints[:6])
    obs.REQUEST_COUNT.labels(endpoint, method, str(response.status_code)).inc()
    return response

# --- UPDATED Core Speech Endpoint Logic ---

def _speech_impl(req: SpeechRequest, base_dir: str, endpoint_path: str):
    if not req.input or not isinstance(req.input, str):
        raise HTTPException(status_code=400, detail="'input' must be a non-empty string")

    fmt = (req.response_format or "wav").lower()
    allowed = {"wav", "pcm", "mp3", "opus", "aac"}
    if fmt not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {req.response_format}")

    # SSE Streaming (Fallback Implementation)
    if (req.stream_format or "").lower() == "sse":
        async def sse_gen():
            import json as _json, base64 as _b64
            from vibevoice_api.audio_utils import float_to_pcm16
            yield f"event: start\ndata: {_json.dumps({'format': 'pcm', 'sample_rate': CONFIG.sample_rate})}\n\n"
            try:
                # Pass all new performance parameters to the streaming function
                async for chunk in synthesize_stream_pcm(
                    root_dir=base_dir, text=req.input, voice=req.voice, 
                    voice_path=req.voice_path, voice_data_b64=req.voice_data,
                    speakers=req.speakers, response_format=fmt, speed=req.speed,
                    seed=req.seed, cfg_scale=req.cfg_scale, ddpm_steps=req.ddpm_steps,
                    use_sampling=req.use_sampling, temperature=req.temperature, top_p=req.top_p,
                    negative_llm_steps_to_cache=req.negative_llm_steps_to_cache,
                    increase_cfg=req.increase_cfg, split_by_newline=req.split_by_newline
                ):
                    pcm_bytes = float_to_pcm16(chunk)
                    b64_data = _b64.b64encode(pcm_bytes).decode()
                    yield f"event: chunk\ndata: {_json.dumps({'type': 'audio_chunk', 'data': b64_data})}\n\n"
            except Exception as e:
                log.error(f"SSE generation error: {e}", exc_info=True)
                yield f"event: error\ndata: {_json.dumps({'error': str(e)})}\n\n"
            yield "event: end\ndata: {}\n\n"
        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    # Regular (non-streaming) generation
    try:
        # Pass all new performance parameters to the synthesis function
        data, content_type = synthesize(
            root_dir=base_dir, text=req.input, voice=req.voice,
            voice_path=req.voice_path, voice_data_b64=req.voice_data,
            speakers=req.speakers, response_format=fmt, speed=req.speed,
            seed=req.seed, cfg_scale=req.cfg_scale, ddpm_steps=req.ddpm_steps,
            use_sampling=req.use_sampling, temperature=req.temperature, top_p=req.top_p,
            negative_llm_steps_to_cache=req.negative_llm_steps_to_cache,
            increase_cfg=req.increase_cfg, split_by_newline=req.split_by_newline,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"synthesis_error: {type(e).__name__}: {e}")

    headers = {"Content-Type": content_type}
    return Response(content=data, media_type=content_type, headers=headers)

@router.post("/audio/speech")
def audio_speech(req: SpeechRequest, request: Request):
    # ADD THIS LINE:
    log.info(f"====> SILLYTAVERN REQUESTED FORMAT: {req.response_format} | STREAMING: {req.stream_format}")
    
    base_dir = os.getcwd()
    return _speech_impl(req, base_dir, request.url.path)

app.include_router(router)
app.include_router(admin_router)

# --- Server Startup Logic (Unchanged) ---

def main(argv: Optional[list[str]] = None) -> None:
    global CONFIG

    parser = argparse.ArgumentParser(description="Run VibeVoice-API server with exl3 backend")
    parser.add_argument("--host", default=CONFIG.host, help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=CONFIG.port, help="Port to run the server on.")
    parser.add_argument("--diffusion-model-path", default=CONFIG.diffusion_model_path, help="Path or Hub ID to the 'no-llm' diffusion model.")
    parser.add_argument("--llm-model-path", default=CONFIG.llm_model_path, help="Path or Hub ID to the 'exl3' quantized LLM model.")
    parser.add_argument("--quant-mode", default=CONFIG.quantization_mode, help="Quantization for diffusion model, e.g., 'bf16' or 'bnb_nf4'.")
    args = parser.parse_args(argv)

    # Create a dictionary of current config values
    current_config = CONFIG.__dict__

    # Update with new command-line args
    current_config.update({
        'host': args.host,
        'port': args.port,
        'diffusion_model_path': args.diffusion_model_path,
        'llm_model_path': args.llm_model_path,
        'quantization_mode': args.quant_mode,
    })

    # Re-initialize CONFIG with the merged values
    config_mod.CONFIG = CONFIG = ServerConfig(**current_config)
    
    # Set environment variables for the tts_engine to pick up
    os.environ["VIBEVOICE_DIFFUSION_MODEL"] = CONFIG.diffusion_model_path
    os.environ["VIBEVOICE_LLM_MODEL"] = CONFIG.llm_model_path
    os.environ["VIBEVOICE_QUANT_MODE"] = CONFIG.quantization_mode

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()