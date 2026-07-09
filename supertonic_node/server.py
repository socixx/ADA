import os
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supertonic import TTS

engine = None
voice_style = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, voice_style
    print("[Supertonic Container] Loading ONNX weights...", flush=True)
    try:
        # Absolutely zero hardware flags. 
        engine = TTS("supertonic-3", auto_download=True)
        voice_style = engine.get_voice_style(voice_name="F1")
        print("✅ Supertonic 3 Node Online (Native CPU Mode)", flush=True)
    except Exception as e:
        print(f"🛑 Model Boot Failure: {e}", flush=True)
    yield

app = FastAPI(title="Ada Supertonic 3 Isolated Core Engine", lifespan=lifespan)

class SpeechRequest(BaseModel):
    input: str
    speed: float = 1.0
    total_steps: int = 8

@app.get("/health")
async def health():
    return {"status": "healthy" if engine is not None else "initializing"}

@app.post("/v1/audio/speech")
async def text_to_speech_stream(request: SpeechRequest, raw_request: Request):
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine offline")
    
    try:
        raw_body = await raw_request.json()
        print(f"\n[Container API Log] Inbound request received:\n{raw_body}", flush=True)
    except Exception:
        pass

    async def audio_stream_generator():
        try:
            wav, duration = await asyncio.to_thread(
                engine.synthesize, 
                request.input, 
                voice_style=voice_style, 
                lang="en",
                speed=request.speed,
                total_steps=request.total_steps
            )
            
            samples = np.array(wav, dtype=np.float32).flatten()
            if len(samples) > 0:
                int16_samples = (samples * 32767.0).astype(np.int16)
                chunk_size = 4096
                for i in range(0, len(int16_samples.tobytes()), chunk_size):
                    yield int16_samples.tobytes()[i:i+chunk_size]
                    await asyncio.sleep(0.001)
                    
        except Exception as e:
            print(f"[Container Stream Error] Synthesis failed: {e}", flush=True)

    return StreamingResponse(audio_stream_generator(), media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)