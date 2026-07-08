import time
import threading
import sounddevice as sd
import numpy as np
import config
import concurrent.futures
import os
import requests
import re
import queue  # Required for the background playback buffer

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

    def __init__(self):
        self.active_engine = getattr(config, "ACTIVE_TTS", "KOKORO")
        print(f"[Voice] Binding to Audio API Engine ({self.active_engine})...")

        self.http_session = requests.Session() 
        self.stop_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # --- THE FIX: DECOUPLED HARDWARE PLAYBACK QUEUE ---
        self.playback_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        
        self.sample_rate = 24000 if self.active_engine == "QWEN_TTS" else 24000

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

    def _playback_loop(self):
        """Dedicated thread that ONLY handles playing audio, freeing the network loop."""
        while True:
            audio_data = self.playback_queue.get()
            if audio_data is None:
                break
            if not self.stop_event.is_set():
                self._dual_write(audio_data)
            self.playback_queue.task_done()

    def _safe_write(self, stream, data):
        try:
            stream.write(data)
        except Exception:
            pass

    def _dual_write(self, audio_data):
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
            
        if self.active_engine == "FISH_AUDIO":
            xp = np.linspace(0, 1, len(audio_data))
            x_new = np.linspace(0, 1, int(len(audio_data) * (48000 / 44100)))
            audio_data_48k = np.interp(x_new, xp, audio_data.flatten()).reshape(-1, 1)
        else:
            audio_data_48k = np.repeat(audio_data, 2, axis=0)

        stereo_data = np.concatenate((audio_data_48k, audio_data_48k), axis=1)
        stereo_data_int16 = np.clip(stereo_data * 32767, -32768, 32767).astype(np.int16)

        futures = []
        if getattr(self, 'vts_stream', None) is not None:
            futures.append(self.executor.submit(self._safe_write, self.vts_stream, stereo_data_int16))
        if getattr(self, 'hardware_stream', None) is not None:
            futures.append(self.executor.submit(self._safe_write, self.hardware_stream, stereo_data_int16))
            
        # This is the line that was causing the bottleneck! 
        # Moving it to the background thread fixes everything.
        if futures:
            concurrent.futures.wait(futures)

    def stop_with_fade(self, audio_queue):
        print("[Voice] 🛑 Interruption: Clearing hardware buffers...")
        self.stop_event.set()
        
        # Clear the incoming sentence queue
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
                audio_queue.task_done()
            except Exception:
                break
        with audio_queue.mutex:
            audio_queue.queue.clear()
            
        # Clear the physical playback queue instantly
        with self.playback_queue.mutex:
            self.playback_queue.queue.clear()

    def reset_stop(self):
        self.stop_event.clear()

    def generate_voice_chunk(self, text: str):
        clean_text = text.strip()
        if not clean_text:
            return None, self.sample_rate, 0.0
        
        if self.active_engine == "KOKORO":
            url = f"http://127.0.0.1:{getattr(config, 'AUDIO_PORT', 8008)}/v1/audio/speech"
            payload = {"input": clean_text, "voice": config.KOKORO_VOICE, "speed": getattr(config, "KOKORO_SPEED", 1.1)}
            
            try:
                response = self.http_session.post(url, json=payload, stream=True, timeout=None)
                response.raise_for_status()
                
                bytes_per_sample = 2
                block_size = int(self.sample_rate * 0.1) 
                block_bytes = block_size * bytes_per_sample

                buffer = b""
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        buffer += chunk
                    
                    while len(buffer) >= block_bytes:
                        if self.stop_event.is_set():
                            buffer = b""
                            break 
                        
                        raw_bytes = buffer[:block_bytes]
                        buffer = buffer[block_bytes:]
                        
                        raw_int16_samples = np.frombuffer(raw_bytes, dtype=np.int16)
                        normalized_float32_samples = raw_int16_samples.astype(np.float32) / 32767.0
                        
                        # Instantly throw to the playback thread instead of blocking
                        self.playback_queue.put(normalized_float32_samples)
                        
                    if self.stop_event.is_set():
                        break

                if buffer and not self.stop_event.is_set():
                    if len(buffer) % 2 != 0:
                        buffer += b"\x00"
                    raw_int16_samples = np.frombuffer(buffer, dtype=np.int16)
                    normalized_float32_samples = raw_int16_samples.astype(np.float32) / 32767.0
                    self.playback_queue.put(normalized_float32_samples)

                return None, self.sample_rate, 0.0
                
            except Exception as e:
                print(f"[Voice Engine Error] Kokoro stream failure: {e}")
                return None, self.sample_rate, 0.0

        # --- QWEN UNTHROTTLED PROGRESSIVE STREAMING CONSUMER ---
        else:
            url = f"http://{config.QWEN_HOST}:{config.QWEN_PORT}/v1/audio/speech"
            payload = {
                "input": clean_text,
                "voice": "ono_anna",
                "speed": 1.0
            }

            try:
                response = self.http_session.post(url, json=payload, stream=True, timeout=None)
                response.raise_for_status()
                
                bytes_per_sample = 2
                block_size = int(self.sample_rate * 0.05)
                block_bytes = block_size * bytes_per_sample

                buffer = b""
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        buffer += chunk
                    
                    while len(buffer) >= block_bytes:
                        if self.stop_event.is_set():
                            buffer = b""
                            break 
                        
                        raw_bytes = buffer[:block_bytes]
                        buffer = buffer[block_bytes:]
                        
                        raw_int16_samples = np.frombuffer(raw_bytes, dtype=np.int16)
                        normalized_float32_samples = raw_int16_samples.astype(np.float32) / 32767.0
                        
                        # Immediately hand the audio chunk off to the playback thread
                        self.playback_queue.put(normalized_float32_samples)
                        
                    if self.stop_event.is_set():
                        break

                if buffer and not self.stop_event.is_set():
                    if len(buffer) % 2 != 0:
                        buffer += b"\x00"
                    raw_int16_samples = np.frombuffer(buffer, dtype=np.int16)
                    normalized_float32_samples = raw_int16_samples.astype(np.float32) / 32767.0
                    self.playback_queue.put(normalized_float32_samples)

                return None, self.sample_rate, 0.0
                
            except Exception as e:
                print(f"[Voice Engine Error] Streaming socket failure: {e}")
                return None, self.sample_rate, 0.0

    def __del__(self):
        try:
            if hasattr(self, 'playback_queue'):
                self.playback_queue.put(None)
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