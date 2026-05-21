import time
import threading
import sounddevice as sd
import numpy as np
import config
import tqdm as real_tqdm
import concurrent.futures
import os
import urllib.request
import asyncio

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
        if not target_name:
            return None
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if target_name.lower() in dev['name'].lower() and dev['max_output_channels'] > 0:
                    return i
        except Exception as e:
            print(f"[Voice] Audio device scan failed: {e}")
        return None

    def _ensure_asset(self, url, filename):
        """Autonomously downloads required model assets if they don't exist."""
        os.makedirs(config.LOCAL_MODELS_DIR, exist_ok=True)
        filepath = os.path.join(config.LOCAL_MODELS_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"\n[Voice] Missing asset detected: '{filename}'")
            print(f"[Voice] Downloading from {url}...")
            try:
                # Add a generic user agent to prevent 403 blocks from some CDNs
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"[Voice] ✅ '{filename}' downloaded successfully.\n")
            except Exception as e:
                print(f"[Voice] 🛑 FATAL: Failed to download '{filename}': {e}")
                os._exit(1)
        return filepath

    def __init__(self):
        self.engine = config.ACTIVE_TTS
        print(f"[Voice] Initializing Native {self.engine} ONNX Engine...")

        self.stop_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        if self.engine == "KOKORO":
            from kokoro_onnx import Kokoro
            from onnxruntime import InferenceSession # <-- ADD THIS
            
            # Auto-Download the v1.0 models
            model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
            voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
            
            kokoro_path = self._ensure_asset(model_url, "kokoro-v1.0.onnx")
            voices_path = self._ensure_asset(voices_url, "voices-v1.0.bin")

            # CRITICAL FIX: Force Kokoro onto the GPU. Do not let it use CPU.
            print("[Voice] Binding Kokoro to CUDAExecutionProvider...")
            inf_sess = InferenceSession(kokoro_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.pipeline = Kokoro.from_session(inf_sess, voices_path)
            
            self.sample_rate = 24000
            self.voice_name = config.KOKORO_VOICE

            vts_id = self._get_device_id(config.VTS_CABLE_DEVICE_NAME)
            hw_id = self._get_device_id(config.HARDWARE_DEVICE_NAME)

            self.vts_stream = None
            self.hardware_stream = None

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

            try:
                self.hardware_stream = sd.OutputStream(
                    samplerate=48000, channels=2, dtype='int16', device=hw_id
                )
                self.hardware_stream.start()
                print(f"[Voice] Hardware Stream bound to: Default/Ears")
            except Exception as e:
                print(f"[Voice] 🛑 Hardware Stream failure: {e}")

            print("[Voice] Pre-compiling and caching Kokoro ONNX graphs with startup warmup prompt...")
            try:
                list(self.pipeline.create_stream("Warmup", voice=self.voice_name, speed=config.KOKORO_SPEED))
                print("[Voice] Kokoro warmup successful.")
            except Exception as e:
                print(f"[Voice] Warmup Warning: {e}")

    def _safe_write(self, stream, data):
        try:
            stream.write(data)
        except Exception:
            pass

    def _dual_write(self, audio_data):
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
            
        audio_data_48k = np.repeat(audio_data, 2, axis=0)
        stereo_data = np.concatenate((audio_data_48k, audio_data_48k), axis=1)

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

        if self.engine == "KOKORO":
            # Wrap the async generator in an asyncio block
            async def process_stream():
                ttfa = 0.0
                stream = self.pipeline.create_stream(clean_text, voice=self.voice_name, speed=config.KOKORO_SPEED)

                # Kokoro-ONNX yields a tuple of (audio_samples, sample_rate)
                async for chunk_info in stream:
                    if self.stop_event.is_set():
                        break

                    if chunk_info is None:
                        continue

                    # CRITICAL FIX: Extract just the audio array from the tuple
                    audio_data_chunk = chunk_info[0] if isinstance(chunk_info, tuple) else chunk_info

                    if ttfa == 0.0:
                        ttfa = time.perf_counter() - start_t

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
                        
                return ttfa

            # Execute the async stream inside the current thread
            try:
                final_ttfa = asyncio.run(process_stream())
                return None, self.sample_rate, final_ttfa
            except Exception as e:
                import traceback
                print(f"[Voice] Stream Generation Error: {e}")
                traceback.print_exc()
                return None, self.sample_rate, 0.0

    def __del__(self):
        try:
            if hasattr(self, 'vts_stream') and self.vts_stream is not None:
                self.vts_stream.stop()
                self.vts_stream.close()
            if hasattr(self, 'hardware_stream') and self.hardware_stream is not None:
                self.hardware_stream.stop()
                self.hardware_stream.close()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass