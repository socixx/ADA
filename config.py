import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Force an absolute, clean Windows path for Docker volumes to avoid mounting anomalies
LOCAL_MODELS_DIR = r"C:\Users\nejle\projects\Ada\models"

RECORD_RATE = 16000
CHUNK_SIZE = 1024
VAD_THRESHOLD = 0.025       

# --- PIPECAT SEMANTIC VAD SETTINGS ---
HF_VAD_REPO = "pipecat-ai/smart-turn-v3"
VAD_MODEL_FILE = "smart-turn-v3.2-gpu.onnx"

SEMANTIC_TURN_THRESHOLD = 0.65  # Confidence (0.0 to 1.0) required to trigger a cutoff
SILENCE_TIMEOUT = 1.5           # Hard fallback limit if Pipecat fails or you walk away

WHISPER_MODEL = "distil-medium.en" 
HF_LLM_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
LLM_MODEL = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# --- TTS ENGINE CONFIGURATION ---
# Options: "KOKORO" or "TTS"
ACTIVE_TTS = "TTS"

TTS_SPEED = 1.15        # Speeds up her speech rate 0.7-2
TTS_QUALITY_STEPS = 5    # Step count for audio generation 5-12, 8 being default

TTS_MODEL = "Supertone/supertonic-3"
TTS_CONTAINER_NAME = "supertonic-tts"
TTS_HOST = "127.0.0.1"
TTS_PORT = 8002
TTS_VRAM_UTIL = 0.25                              # Allocated execution boundaries
SENTENCE_WINDOW = 1

# Kokoro Baseline Settings
KOKORO_VOICE = "af_bella"
KOKORO_SPEED = 1.25

# Standardized Timbre Reference Assets
TTS_REF_AUDIO = "assets/ada_ref.wav"
TTS_REF_TEXT = "Hello. I am your real-time vocal assistant. Let's ensure this pipeline runs smoothly."

# --- GLOBAL DOCKER AND CACHE CONFIGURATIONS ---
HF_CACHE_DIR = r"C:\Users\nejle\projects\Ada\.cache"
LLM_CONTAINER_NAME = "vllm-llama"
LLM_PORT = 8000                                    # Connects the Brain class directly back to your Llama server endpoint

# --- VISION TOGGLE ---
ENABLE_VISION = False  
VISION_CONTAINER_NAME = "vllm-qwen-vision"
VISION_PORT = 8005

# --- AUDIO ROUTING ---
VTS_CABLE_DEVICE_NAME = "CABLE Input" 
HARDWARE_DEVICE_NAME = None