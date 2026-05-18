import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RECORD_RATE = 16000
CHUNK_SIZE = 1024
VAD_THRESHOLD = 0.025       

# --- PIPECAT SEMANTIC VAD SETTINGS ---
SEMANTIC_VAD_MODEL = "pipecat-ai/smart-turn-v2"
SEMANTIC_TURN_THRESHOLD = 0.65  # Confidence (0.0 to 1.0) required to trigger a cutoff
SILENCE_TIMEOUT = 1.5          # Hard fallback limit if Pipecat fails or you walk away

WHISPER_MODEL = "distil-medium.en" 
LLM_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
LLM_QUANTIZATION = "none" 

ACTIVE_TTS = "KOKORO" 
KOKORO_VOICE = "af_bella"
KOKORO_SPEED = 1.1
KOKORO_LANG = "a"

# --- AUDIO ROUTING ---
VTS_CABLE_DEVICE_ID = 18     # Replace with the ID of 'CABLE Input'
HARDWARE_DEVICE_ID = None   # None = Default Windows Output (For your ears)