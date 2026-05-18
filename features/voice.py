import time
import threading
import sounddevice as sd
import numpy as np
import torch
import config
import tqdm as real_tqdm
import concurrent.futures

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
        
        # Thread pool to ensure perfectly synced dual-audio playback
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        if self.engine == "KOKORO":
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=config.KOKORO_LANG, repo_id='hexgrad/Kokoro-82M', device='cuda')
            self.sample_rate = 24000
            self.voice_name = config.KOKORO_VOICE

            # --- DUAL HARDWARE STREAMS ---
            try:
                self.vts_stream = sd.OutputStream(
                    samplerate=self.sample_rate, channels=1, dtype='float32',
                    device=config.VTS_CABLE_DEVICE_ID
                )
                self.vts_stream.start()
                
                self.hardware_stream = sd.OutputStream(
                    samplerate=self.sample_rate, channels=1, dtype='float32',
                    device=config.HARDWARE_DEVICE_ID
                )
                self.hardware_stream.start()
                print(f"[Voice] Dual hardware streams opened. (VTS: {config.VTS_CABLE_DEVICE_ID}, Hardware: {config.HARDWARE_DEVICE_ID or 'Default'})")
            except Exception as e:
                print(f"[Voice] 🛑 Failed to open dual streams: {e}")

            print("[Voice] Pre-compiling and caching Kokoro graphs with startup warmup prompt...")
            try:
                warmup_generator = self.pipeline("Warmup", voice=self.voice_name, speed=config.KOKORO_SPEED)
                for gs, ps, audio_data_chunk in warmup_generator:
                    pass
                print("[Voice] Kokoro warmup successful.")
            except Exception as e:
                print(f"[Voice] Kokoro warmup warning: {e}")

    def _safe_write(self, stream, data):
        """Silently catches errors if a device disconnects mid-stream."""
        try:
            stream.write(data)
        except Exception:
            pass

    def _dual_write(self, audio_data):
        """Fires the audio array to both streams simultaneously to prevent desync."""
        f1 = self.executor.submit(self._safe_write, self.vts_stream, audio_data)
        f2 = self.executor.submit(self._safe_write, self.hardware_stream, audio_data)
        concurrent.futures.wait([f1, f2])

    def stop_with_fade(self, audio_queue):
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

                block_size = int(self.sample_rate * 0.1) 
                
                for i in range(0, len(audio_data), block_size):
                    if self.stop_event.is_set():
                        trail_samples = int(self.sample_rate * 0.5) 
                        remaining = audio_data[i : i + trail_samples]
                        
                        if len(remaining) > 0:
                            abs_audio = np.abs(remaining).flatten()
                            window = int(self.sample_rate * 0.015) 
                            break_point = len(remaining)
                            
                            for j in range(0, len(abs_audio) - window, int(window/2)):
                                rms = np.mean(abs_audio[j : j + window])
                                if rms < 0.015: 
                                    break_point = j + int(window/2)
                                    break
                            
                            syllable_finish = remaining[:break_point]
                            self._dual_write(syllable_finish)
                            
                            tail_len = min(int(self.sample_rate * 0.05), len(remaining) - break_point)
                            if tail_len > 0:
                                tail = remaining[break_point : break_point + tail_len]
                                fade = (np.linspace(1.0, 0.0, tail_len, dtype=np.float32) ** 2).reshape(-1, 1)
                                self._dual_write(tail * fade)
                        
                        break 
                    
                    end_idx = min(i + block_size, len(audio_data))
                    self._dual_write(audio_data[i : end_idx])

                if self.stop_event.is_set():
                    break 

            return None, self.sample_rate, ttfa

    def __del__(self):
        try:
            if hasattr(self, 'vts_stream'):
                self.vts_stream.stop()
                self.vts_stream.close()
            if hasattr(self, 'hardware_stream'):
                self.hardware_stream.stop()
                self.hardware_stream.close()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass