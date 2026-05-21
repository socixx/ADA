import os
import io
import asyncio
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Response
from pydantic import BaseModel
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

app = FastAPI()

MODELS_DIR = "/models"
WHISPER_MODEL = "distil-medium.en"

print("[Audio Node] Loading Faster-Whisper...")
whisper_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="int8_float16", download_root=MODELS_DIR)

# Optimize ONNX Runtime for Docker / WSL2 memory handoffs
options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 4 

print("[Audio Node] Loading Kokoro-ONNX...")
kokoro_path = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
voices_path = os.path.join(MODELS_DIR, "voices-v1.0.bin")
inf_sess = InferenceSession(kokoro_path, sess_options=options, providers=["CUDAExecutionProvider"])
kokoro_pipeline = Kokoro.from_session(inf_sess, voices_path)

# --- CRITICAL FIX: GPU COLD-START WARMUP ---
print("[Audio Node] Warming up CUDA execution graphs. Please wait...")
async def warmup_engine():
    try:
        stream = kokoro_pipeline.create_stream("Engine warmup sequence initiated.", voice="af_bella", speed=1.1)
        async for _ in stream:
            pass # We don't care about the audio, just force the GPU to compile the math
        print("[Audio Node] ✅ Warmup complete. Engine ready at 0ms latency.")
    except Exception as e:
        print(f"[Audio Node] Warmup warning: {e}")

asyncio.run(warmup_engine())
# -------------------------------------------

class SpeechRequest(BaseModel):
    input: str
    voice: str = "af_bella"
    speed: float = 1.1

@app.post("/v1/audio/transcriptions")
def transcribe(file: UploadFile = File(...)):
    """Receives a WAV file buffer, processes it, and returns text."""
    audio_bytes = file.file.read()
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    segments, _ = whisper_model.transcribe(
        data, beam_size=1, language="en", condition_on_previous_text=False, temperature=0.0
    )
    text = "".join([segment.text for segment in segments]).strip()
    return {"text": text}

@app.post("/v1/audio/speech")
def generate_speech(req: SpeechRequest):
    """Generates the TTS array and returns a single binary blast."""
    async def get_audio():
        stream = kokoro_pipeline.create_stream(req.input, voice=req.voice, speed=req.speed)
        chunks = []
        async for chunk_info in stream:
            if chunk_info is not None:
                audio = chunk_info[0] if isinstance(chunk_info, tuple) else chunk_info
                chunks.append(audio.astype(np.float32))
        
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    final_audio = asyncio.run(get_audio())
    return Response(content=final_audio.tobytes(), media_type="application/octet-stream")