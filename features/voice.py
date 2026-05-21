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
    def _get_device_id(self, target_name):
        """Scans Windows audio devices and returns the ID of a matching name."""
        if not target_name:
            return None
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                # Look for the name and ensure it's actually an output device
                if target_name.lower() in dev['name'].lower() and dev['max_output_channels'] > 0:
                    return i
        except Exception as e:
            print(f"[Voice] Audio device scan failed: {e}")
        return None

    def __init__(self):
        self.engine = config.ACTIVE_TTS
        print(f"[Voice] Initializing Native {self.engine} Engine into VRAM...")

        self.stop_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        if self.engine == "KOKORO":
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=config.KOKORO_LANG, repo_id='hexgrad/Kokoro-82M', device='cuda')
            self.sample_rate = 24000
            self.voice_name = config.KOKORO_VOICE

            vts_id = self._get_device_id(config.VTS_CABLE_DEVICE_NAME)
            hw_id = self._get_device_id(config.HARDWARE_DEVICE_NAME)

            self.vts_stream = None
            self.hardware_stream = None

            # --- VTS STREAM (ISOLATED) ---
            try:
                if vts_id is not None:
                    self.vts_stream = sd.OutputStream(
                        samplerate=48000, channels=2, dtype='int16', device=vts_id
                    )
                    self.vts_stream.start()
                    print(f"[Voice] VTS Stream bound to: '{config.VTS_CABLE_DEVICE_NAME}' (ID: {vts_id})")
                else:
                    print(f"[Voice] ⚠️ Could not find audio cable matching '{config.VTS_CABLE_DEVICE_NAME}'")
            except Exception as e:
                print(f"[Voice] 🛑 VTS Stream blocked by Windows: {e}")

            # --- HARDWARE STREAM / HEADPHONES (ISOLATED) ---
            try:
                self.hardware_stream = sd.OutputStream(
                    samplerate=48000, channels=2, dtype='int16', device=hw_id
                )
                self.hardware_stream.start()
                print(f"[Voice] Hardware Stream bound to: Default/Ears")
            except Exception as e:
                print(f"[Voice] 🛑 Hardware Stream failure: {e}")

            print("[Voice] Pre-compiling and caching Kokoro graphs with startup warmup prompt...")
            try:
                for _ in self.pipeline("Warmup", voice=self.voice_name, speed=config.KOKORO_SPEED):
                    pass
                print("[Voice] Kokoro warmup successful.")
            except Exception as e:
                pass

    def _safe_write(self, stream, data):
        """Silently catches errors if a device disconnects mid-stream."""
        try:
            stream.write(data)
        except Exception:
            pass

    def _dual_write(self, audio_data):
        """Fires the audio array to both streams simultaneously. Fails gracefully if streams are dead."""
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
            
        audio_data_48k = np.repeat(audio_data, 2, axis=0)
        stereo_data = np.concatenate((audio_data_48k, audio_data_48k), axis=1)

        # Convert float32 (-1.0 to 1.0) into PCM int16 (-32768 to 32767) to satisfy Windows Drivers
        stereo_data_int16 = np.clip(stereo_data * 32767, -32768, 32767).astype(np.int16)

        futures = []
        if hasattr(self, 'vts_stream') and self.vts_stream is not None:
            futures.append(self.executor.submit(self._safe_write, self.vts_stream, stereo_data_int16))
            
        if hasattr(self, 'hardware_stream') and self.hardware_stream is not None:
            futures.append(self.executor.submit(self._safe_write, self.hardware_stream, stereo_data_int16))
            
        if futures:
            concurrent.futures.wait(futures)

    def stop_with_fade(self, audio_queue):
        print("[Voice] 🛑 Interruption: Trailing off audio naturally...")
        self.stop_event.set()

        # CRITICAL FIX: Properly drain the queue. Using .clear() causes .join() deadlocks!
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
                audio_queue.task_done()
            except Exception:
                break

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