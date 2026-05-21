import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODELS_DIR = os.path.join(BASE_DIR, "models")

RECORD_RATE = 16000
CHUNK_SIZE = 1024
VAD_THRESHOLD = 0.025       

# --- PIPECAT SEMANTIC VAD SETTINGS ---
HF_VAD_REPO = "pipecat-ai/smart-turn-v3"
VAD_MODEL_FILE = "smart-turn-v3.2-gpu.onnx"

SEMANTIC_TURN_THRESHOLD = 0.65  # Confidence (0.0 to 1.0) required to trigger a cutoff
SILENCE_TIMEOUT = 1.5          # Hard fallback limit if Pipecat fails or you walk away

WHISPER_MODEL = "distil-medium.en" 
HF_LLM_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
LLM_MODEL = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LLM_QUANTIZATION = "none" 

ACTIVE_TTS = "KOKORO" 
KOKORO_VOICE = "af_bella"
KOKORO_SPEED = 1.25
KOKORO_LANG = "a"

# --- AUDIO ROUTING ---
VTS_CABLE_DEVICE_NAME = "CABLE Input"  
HARDWARE_DEVICE_NAME = None            # None = Default Windows Output)

# --- VLLM DOCKER ORCHESTRATION CONFIG ---
LAUNCH_VLLM_CONTAINERS = True
HF_CACHE_DIR = r"C:\Users\nejle\.cache\huggingface"
LOCAL_MODELS_DIR = os.path.join(BASE_DIR, "models")

# Text Engine Node Allocation (45% VRAM)
LLM_CONTAINER_NAME = "vllm-llama"
LLM_PORT = 8000
LLM_VRAM_UTIL = 0.30

# Vision Engine Node Allocation (25% VRAM)
VISION_CONTAINER_NAME = "vllm-qwen-vision"
VISION_PORT = 8005
VISION_VRAM_UTIL = 0.23
VISION_CONTEXT = 2048       # Shrunk from 4096 since Vision is stateless

# Add this under your Vision Engine config
AUDIO_CONTAINER_NAME = "audio-engine-node"
AUDIO_PORT = 8008