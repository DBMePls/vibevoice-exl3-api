# 🎙️ VibeVoice-EXL3 API

> **An insanely fast, low-VRAM, OpenAI-compatible REST API for VibeVoice, powered by ExLlamaV3.**

This is a heavily optimized fork of the original VibeVoice API. By decoupling the LLM generation phase from the diffusion process and running it through a custom **ExLlamaV3** backend, this API drastically reduces VRAM requirements and significantly accelerates time-to-first-audio (TTFA) and overall generation speed.

It functions as a drop-in replacement for OpenAI's `audio.speech.create` endpoint, meaning it works instantly with existing frontend UIs, Chatbots, and SDKs.

---

## ✨ Key Features
* 🚀 **ExLlamaV3 Backend:** The autoregressive LLM component runs on EXL3, utilizing highly optimized CUDA kernels for lightning-fast token generation.
* 💾 **Low VRAM Footprint:** Fits much larger VibeVoice models into consumer GPUs (e.g., 8-bit quantization) without Out-Of-Memory errors.
* 🌐 **OpenAI SDK Compatible:** Drop-in replacement for the official OpenAI Python/JS SDKs.
* ⚡ **True Streaming:** Supports Server-Sent Events (SSE) and native binary streaming (chunk-by-chunk playback before generation finishes).
* 🎭 **Zero-Shot Voice Cloning:** Clone voices on the fly by passing a path or base64 string to a reference `.wav` file.
* 🎛️ **Advanced Generation Controls:** Tune Dynamic CFG, temperature, top_p, and diffusion steps directly via API request payloads.
* 💻 **Built-in Web Console:** Includes a fully functional HTML streaming console for testing voices and testing generation parameters.

---

## 🤝 Acknowledgements & Credits

This project stands on the shoulders of giants. Massive thanks to the open-source community:
* **[Mozer](https://github.com/Mozer/exllamav3)** - For reverse-engineering the VibeVoice architecture and porting the LLM component to the EXL3 engine. (Check out their [ComfyUI Node](https://github.com/mozer/comfyUI-vibevoice-exl3)!).
* **[Turboderp](https://github.com/turboderp-org/exllamav3)** - The creator of ExLlamaV3, without which this level of local LLM performance wouldn't be possible.
* **[VibeVoice-Community](https://github.com/vibevoice-community/VibeVoice-API)** - For the original OpenAI-compatible API skeleton this fork is based on.

---

## 📦 Installation

Because this API relies on ExLlamaV3, you need a proper CUDA/C++ build environment on your system (e.g., Visual Studio C++ Build Tools on Windows, or `build-essential` on Linux).

### 1. Install the Custom ExLlamaV3 Backend
You must install the specialized VibeVoice fork of ExLlamaV3 first.
```bash
pip install git+https://github.com/DontBlameMep/exllamav3-vibevoice.git
```
*(Note: This compiles CUDA kernels and may take several minutes).*

### 2. Install this API
```bash
git clone https://github.com/DontBlameMep/vibevoice-exl3-api.git
cd vibevoice-exl3-api

# Install the API requirements
pip install -e .
```

*(Optional) If you want to output non-WAV formats like `mp3`, `opus`, or `aac`, ensure you have `ffmpeg` installed and added to your system PATH.*

---

## 🧠 Model Weights

Because this implementation uses ExLlamaV3, the standard VibeVoice `.safetensors` files won't work natively. The model has been cleanly split into two parts:
1. **The LLM Component** (Quantized to EXL3)
2. **The Diffusion Component** (No-LLM, keeping the acoustic tokenizers and UNet)

By default, the server will attempt to download these pre-split models from HuggingFace on the first run:
* **LLM:** `tensorbanana/vibevoice-7b-exl3-8bit`
* **Diffusion:** `tensorbanana/vibevoice-7b-no-llm-bf16`

*(If you download these manually, you can point the server to your local folders via arguments).*

---

## 🚀 Usage & Configuration

### Starting the Server
Start the API server by running:
```bash
python -m vibevoice_api.server --port 8000
```

**Optional Arguments:**
* `--host 0.0.0.0` (Expose to local network)
* `--port 8000` (Change listening port)
* `--diffusion-model-path <path>` (Path to the No-LLM diffusion folder)
* `--llm-model-path <path>` (Path to the EXL3 quantized folder)

### Accessing the Web Console
Once the server is running, you can access the built-in testing interface from your browser:
* **Streaming Console:** `http://127.0.0.1:8000/v1/web/console.html`

---

## 📡 API Examples

All routes are mounted under `/v1` to match OpenAI's structure.

### 🐍 Python (Using the Official OpenAI SDK)
```python
from openai import OpenAI

# Point the client to your local server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1", 
    api_key="sk-no-key-required"
)

response = client.audio.speech.create(
    model="vibevoice", # Ignored by backend (loads defaults)
    voice="Alice",     # Name of reference voice in /demo/voices/
    input="This generation is incredibly fast thanks to ExLlamaV3!",
    response_format="wav",
)

with open("output.wav", "wb") as f:
    f.write(response.read())
```

### 💻 cURL (Standard HTTP Request)
```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "voice": "Alice",
    "input": "This is a direct API call test.",
    "response_format": "mp3"
  }' \
  --output test.mp3
```

### 🎭 Custom Voice Cloning (Zero-Shot)
You can clone a voice on the fly without restarting the server by passing a path to a reference `.wav` file in the `extra_body`.
```python
response = client.audio.speech.create(
    model="vibevoice",
    voice="ignored", # Ignored when using voice_path
    input="I am speaking with a brand new cloned voice.",
    response_format="wav",
    extra_body={
        "voice_path": "C:/path/to/my/custom_voice.wav"
    }
)
```

*(You can also manage permanent voices by dropping them in the `demo/voices/` folder, or mapping them in `voice_map.yaml`).*

---

## ⚙️ Advanced Generation Tuning
You can pass additional kwargs inside `extra_body` (in Python) or directly in your JSON payload to fine-tune the EXL3 and Diffusion steps:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ddpm_steps` | int | `16` | Number of diffusion steps. Lower = faster, Higher = better quality. |
| `cfg_scale` | float | `1.3` | Classifier-free guidance. Controls adherence to the reference voice. |
| `temperature` | float | `0.95` | LLM temperature. Alters the pacing/inflection of the generated speech. |
| `increase_cfg` | bool | `false` | Experimental: Dynamically boosts CFG during the first 50% of steps for more emotion. |
| `split_by_newline`| bool | `false` | Automatically splits long text by paragraphs to prevent context degradation. |

---

## 📜 License
This project is released under the **GNU General Public License v3.0 License**. 