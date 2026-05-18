# Project Ada: Ultra-Low-Latency Local AI VTuber Architecture

## 1. Design Philosophy
To achieve a true "live" conversational feel with an AI VTuber, traditional synchronous API calls and basic frameworks must be avoided. The core design principles for Ada are:
* **100% Local Execution:** No cloud dependencies, ensuring absolute privacy and zero network latency.
* **Asynchronous Pipeline-Parallelism:** Modules do not wait for each other. Data flows continuously through memory-backed async queues.
* **Raw Binary Streaming:** Avoid JSON overhead for high-bandwidth data like audio; stream raw PCM bytes directly to the frontend.

---

## 2. The Core Technology Stack

### A. The Ears (ASR - Automatic Speech Recognition)
* **Gatekeeper:** `Silero VAD` (Voice Activity Detection). Detects speech start/stop in milliseconds to avoid waiting for artificial silence timeouts.
* **Transcriber:** `Faster-Whisper` (configured with `CTranslate2`). Runs on GPU with `float16` or `int8_float16` quantization for near-instant transcription.

### B. The Brain (LLM & Orchestrator)
* **Inference Engine:** `vLLM` or `SGLang`. These highly optimized servers use PagedAttention and FlashAttention to maximize tokens-per-second and minimize Time-to-First-Token (TTFT).
* **Model:** An 8B parameter instruction-tuned model (e.g., `Llama-3.1-8B-Instruct` or `Mistral-7B-Instruct`), quantized via AWQ or GPTQ to fit entirely within VRAM.

### C. The Voice (TTS - Text-to-Speech)
* **Engine:** `StyleTTS2` or `Kokoro-82M`. 
* **Why:** These engines provide hyper-realistic human prosody with extremely low inference latency (Real-Time Factor). They can generate a sentence's audio in under 50ms.
* **Lip-Sync Data:** The TTS engine must output both the audio tensor and precise phoneme/viseme timestamps for accurate mouth movements.

### D. The Body (Live2D Rigging & Frontend)
* **Environment:** Node.js / HTML5 utilizing `PixiJS` and the official `pixi-live2d-display` library.
* **Integration:** Runs in a local web browser, captured as a transparent overlay in OBS Studio.
* **Control:** Phonemes from the TTS are mapped to Live2D parameters (`ParamMouthForm`, `ParamMouthOpenY`) in real-time.

---

## 3. High-Performance Pipeline Blueprint

The system uses an Asynchronous Producer-Consumer Pipeline built with Python's `asyncio` and WebSockets/ZeroMQ for Inter-Process Communication (IPC).

```text
[ Microphone Input ] ──(Audio Stream)──> [ Async Queue 1: VAD & ASR ]
                                                    │
                                             (Text String)
                                                    ▼
[ Local LLM Engine (vLLM) ] <─────────── [ Async Queue 2: Brain Router ]
         │
  (Token Stream)
         ▼
[ Async Queue 3: Sentence Chunking ] 
         │
 (Complete Clauses)
         ▼
[ Local TTS Engine (StyleTTS2) ]
         │
 (Raw PCM Audio Bytes + Viseme Data)
         ▼
[ WebSocket Server ] ──(Binary Stream)──> [ Live2D Browser Frontend (PixiJS) ]