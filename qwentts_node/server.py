import os
import torch
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_PATH = "/app/model"
model = None

def apply_anti_clipping_fade(audio_samples: np.ndarray, fade_len: int = 200) -> np.ndarray:
    if len(audio_samples) > fade_len:
        fade_window = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        audio_samples[-fade_len:] *= fade_window
    return audio_samples

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("[Node Engine] Loading weights cleanly onto GPU...", flush=True)
    try:
        from faster_qwen3_tts import FasterQwen3TTS
        model = FasterQwen3TTS.from_pretrained(MODEL_PATH)
        
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(torch.float16)
        elif hasattr(model, "to"):
            model.to(torch.float16)
            
        # Quick warmup generation pass to pre-compile execution matrices
        generator = model.generate_custom_voice_streaming(
            text="Warmup pass.", language="English", speaker="ono_anna", chunk_size=8
        )
        next(generator, (None, None, None))
        print("✅ Production Faster-Qwen3-TTS Streaming Engine Active!", flush=True)
    except Exception as e:
        print(f"🛑 Engine Boot Failure: {e}", flush=True)
    yield

app = FastAPI(title="Ada True Async Streaming Engine", lifespan=lifespan)

class SpeechRequest(BaseModel):
    input: str
    voice: str = "ono_anna"
    speed: float = 1.0

@app.post("/v1/audio/speech")
async def text_to_speech_stream(request: SpeechRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Engine offline")
    
    async def audio_stream_generator():
        try:
            target_speaker = request.voice.lower() if request.voice else "ono_anna"
            
            # Use to_thread to safely run the synchronous generator inside FastAPI's async environment
            def get_generator():
                return model.generate_custom_voice_streaming(
                    text=request.input,
                    language="English",
                    speaker=target_speaker,
                    chunk_size=8,
                    non_streaming_mode=False # NATIVE STEP-BY-STEP GENERATION UNLOCKED
                )
            
            generator = await asyncio.to_thread(get_generator)
            
            while True:
                # Pull raw fragments off the model pipeline one single step at a time
                audio_chunk, sr, timing = await asyncio.to_thread(next, generator, (None, None, None))
                if audio_chunk is None:
                    break
                
                if isinstance(audio_chunk, torch.Tensor):
                    samples = audio_chunk.cpu().numpy().astype(np.float32).flatten()
                else:
                    samples = np.array(audio_chunk, dtype=np.float32).flatten()

                if len(samples) > 0:
                    int16_chunk = (samples * 32767.0).astype(np.int16)
                    yield int16_chunk.tobytes()
                    
        except StopIteration:
            pass
        except Exception as e:
            print(f"[Stream Error] Generation pipeline exception: {e}", flush=True)

    return StreamingResponse(audio_stream_generator(), media_type="audio/pcm")

@app.get("/health")
async def health():
    return {"status": "healthy" if model else "offline"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)