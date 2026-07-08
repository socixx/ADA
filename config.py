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
# Options: "KOKORO" or "QWEN_TTS"
ACTIVE_TTS = "QWEN_TTS"

# Kokoro Baseline Settings
KOKORO_VOICE = "af_bella"
KOKORO_SPEED = 1.25

# Qwen3-TTS-12Hz-1.7B-Base Native vLLM-OpenAI Stack Configurations
QWEN_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"      # Base is for zero shot - CustomVoice is for tibre made voices - VoiceDesign is for text descriped voice sound
QWEN_CONTAINER_NAME = "vllm-qwen-tts"
QWEN_HOST = "127.0.0.1"
QWEN_PORT = 8002
QWEN_VRAM_UTIL = 0.25                              # Locks engine VRAM consumption tightly within a 4-6GB footprint
# Higher numbers = more emotional consistency, Lower numbers (e.g., 1 or 2) = faster response times
QWEN_SENTENCE_WINDOW = 4

# --- QWEN3-TTS VOICE CLONING REFERENCE ASSETS ---
QWEN_REF_AUDIO = "assets/ada_ref.wav"
QWEN_REF_TEXT = "Hello. I am your real-time vocal assistant. Let's ensure this pipeline runs smoothly."

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