import time
import threading
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

        self.stop_event = threading.Event()

        if self.engine == "KOKORO":
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=config.KOKORO_LANG, device='cuda')
            self.sample_rate = 24000
            self.voice_name = config.KOKORO_VOICE

            self.output_stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=1, dtype='float32'
            )
            self.output_stream.start()
            print("[Voice] Persistent audio output hardware stream opened and initialized.")

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

    def stop_with_fade(self, audio_queue):
        """
        Signals the audio worker to gracefully trail off. 
        We no longer write to the stream here to avoid thread race conditions.
        """
        print("[Voice] 🛑 Interruption: Trailing off audio naturally...")
        self.stop_event.set()

        with audio_queue.mutex:
            audio_queue.queue.clear()

    def reset_stop(self):
        self.stop_event.clear()

    def generate_voice_chunk(self, text: str):
        start_t = time.perf_counter()

        clean_text = text.strip()
        if not clean_text:
            return None, self.sample_rate, 0.0

        ttfa = 0.0

        if self.engine == "KOKORO":
            generator = self.pipeline(clean_text, voice=self.voice_name, speed=config.KOKORO_SPEED)

            for gs, ps, audio_data_chunk in generator:
                if self.stop_event.is_set():
                    break

                if audio_data_chunk is None:
                    continue

                if ttfa == 0.0:
                    ttfa = time.perf_counter() - start_t

                if isinstance(audio_data_chunk, torch.Tensor):
                    audio_data_chunk = audio_data_chunk.cpu().numpy()

                vol_threshold = 0.005
                active_frames = np.where(np.abs(audio_data_chunk) > vol_threshold)[0]
                if len(active_frames) > 0:
                    end_idx = min(
                        len(audio_data_chunk),
                        active_frames[-1] + int(self.sample_rate * 0.025) 
                    )
                    audio_data_chunk = audio_data_chunk[:end_idx]

                audio_data = audio_data_chunk.astype('float32')
                if audio_data.ndim == 1:
                    audio_data = audio_data.reshape(-1, 1)

                # --- NEW: Syllable-Aware Exponential Trail-Off ---
                block_size = int(self.sample_rate * 0.1) # 100ms blocks
                
                for i in range(0, len(audio_data), block_size):
                    if self.stop_event.is_set():
                        # Grab the next 500ms of audio to analyze for a syllable break
                        trail_samples = int(self.sample_rate * 0.5) 
                        remaining = audio_data[i : i + trail_samples]
                        
                        if len(remaining) > 0:
                            # 1. Envelope Detection: Find a natural dip in amplitude
                            abs_audio = np.abs(remaining).flatten()
                            window = int(self.sample_rate * 0.015) # 15ms scanning window
                            break_point = len(remaining)
                            
                            # Scan forward to find where the audio energy drops near zero
                            for j in range(0, len(abs_audio) - window, int(window/2)):
                                rms = np.mean(abs_audio[j : j + window])
                                if rms < 0.015: # Acoustic trough threshold
                                    break_point = j + int(window/2)
                                    break
                            
                            # 2. Play up to the break point at normal volume
                            syllable_finish = remaining[:break_point]
                            self.output_stream.write(syllable_finish)
                            
                            # 3. Apply an exponential fade to a short tail to kill any pop/click
                            tail_len = min(int(self.sample_rate * 0.05), len(remaining) - break_point) # 50ms tail
                            if tail_len > 0:
                                tail = remaining[break_point : break_point + tail_len]
                                # Quadratic decay (**2) sounds much more natural than linear
                                fade = (np.linspace(1.0, 0.0, tail_len, dtype=np.float32) ** 2).reshape(-1, 1)
                                self.output_stream.write(tail * fade)
                        
                        break # Break out of the block loop
                    
                    end_idx = min(i + block_size, len(audio_data))
                    self.output_stream.write(audio_data[i : end_idx])

                if self.stop_event.is_set():
                    break # Break out of the Kokoro generator loop entirely

            return None, self.sample_rate, ttfa

        elif self.engine == "CHATTERBOX":
            with torch.inference_mode():
                full_wav = self.model.generate(clean_text)

            ttfa = time.perf_counter() - start_t

            if isinstance(full_wav, torch.Tensor):
                audio_data = full_wav.squeeze().cpu().numpy()
            else:
                audio_data = np.asarray(full_wav)

            if audio_data.size > 0 and not self.stop_event.is_set():
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