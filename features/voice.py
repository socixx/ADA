import time
import sounddevice as sd
import numpy as np
import torch
import config
import tqdm as real_tqdm

class NoOpTqdm(real_tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs['disable'] = True
        super().__init__(*args, **kwargs)

real_tqdm.tqdm = NoOpTqdm
try:
    import tqdm.auto
    tqdm.auto.tqdm = NoOpTqdm
except Exception:
    pass

class Voice:
    def __init__(self):
        self.engine = config.ACTIVE_TTS
        print(f"[Voice] Initializing Native {self.engine} Engine into VRAM...")

        if self.engine == "KOKORO":
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=config.KOKORO_LANG, device='cuda')
            self.sample_rate = 24000
            self.voice_name = config.KOKORO_VOICE 
            
            self.output_stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32')
            self.output_stream.start()
            print(f"[Voice] Persistent audio output hardware stream opened and initialized.")
            
            print("[Voice] Pre-compiling and caching Kokoro graphs with startup warmup prompt...")
            try:
                warmup_generator = self.pipeline("Warmup", voice=self.voice_name, speed=config.KOKORO_SPEED)
                for gs, ps, audio_data_chunk in warmup_generator:
                    pass
                print("[Voice] Kokoro warmup successful. Kernel execution graph optimized and cached.")
            except Exception as e:
                print(f"[Voice] Kokoro warmup warning: {e}")
            
        elif self.engine == "CHATTERBOX":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
            self.sample_rate = self.model.sr
            print(f"[Voice] Chatterbox Turbo Engine active. Sample rate: {self.sample_rate}Hz")

    def generate_voice_chunk(self, text: str):
        start_t = time.perf_counter()
        
        clean_text = text.strip()
        if not clean_text:
            return None, self.sample_rate, 0.0

        ttfa = 0.0
        
        if self.engine == "KOKORO":
            generator = self.pipeline(clean_text, voice=self.voice_name, speed=config.KOKORO_SPEED)
            for gs, ps, audio_data_chunk in generator:
                if audio_data_chunk is None: 
                    continue
                    
                if ttfa == 0.0:
                    ttfa = time.perf_counter() - start_t
                
                if isinstance(audio_data_chunk, torch.Tensor):
                    audio_data_chunk = audio_data_chunk.cpu().numpy()
                
                # --- DYNAMIC WAVEFORM TRIMMING ---
                # Scan the audio array for actual vocal frequencies. 
                # Slice off Kokoro's baked-in trailing silence while keeping the pitch contour intact.
                vol_threshold = 0.005 
                active_frames = np.where(np.abs(audio_data_chunk) > vol_threshold)[0]
                if len(active_frames) > 0:
                    # Keep audio up to the last audible frame + a tiny 25ms safety pad
                    end_idx = min(len(audio_data_chunk), active_frames[-1] + int(self.sample_rate * 0.025))
                    audio_data_chunk = audio_data_chunk[:end_idx]
                    
                if audio_data_chunk.ndim == 1:
                    audio_data_chunk = audio_data_chunk.reshape(-1, 1)
                    
                self.output_stream.write(audio_data_chunk.astype('float32'))
                
            return None, self.sample_rate, ttfa

        elif self.engine == "CHATTERBOX":
            with torch.inference_mode():
                full_wav = self.model.generate(clean_text)
            
            ttfa = time.perf_counter() - start_t
            
            if isinstance(full_wav, torch.Tensor):
                audio_data = full_wav.squeeze().cpu().numpy()
            else:
                audio_data = np.asarray(full_wav)
                
            if audio_data.size > 0:
                if audio_data.ndim == 1:
                    audio_data = audio_data.reshape(-1, 1)
                
                with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
                    stream.write(audio_data.astype('float32'))
                    
            return None, self.sample_rate, ttfa

    def __del__(self):
        if hasattr(self, 'output_stream'):
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception:
                pass