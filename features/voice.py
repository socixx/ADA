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
import json

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
        
        # Sample rate sets dynamically based on the active engine selection branch
        if self.active_engine == "KOKORO":
            self.sample_rate = 24000
        else:
            self.sample_rate = 44100 # Supertonic 3 outputs 44.1kHz natively

        vts_id = self._get_device_id(config.VTS_CABLE_DEVICE_NAME)

        self.vts_stream = None

        try:
            if vts_id is not None:
                self.vts_stream = sd.OutputStream(
                    samplerate=self.sample_rate, channels=2, dtype='int16', device=vts_id
                )
                self.vts_stream.start()
                print(f"[Voice] VTS Stream bound to: '{config.VTS_CABLE_DEVICE_NAME}' (ID: {vts_id})")
        except Exception as e:
            print(f"[Voice] 🛑 VTS Stream blocked by Windows: {e}")

    def _playback_loop(self):
        """Dedicated thread that ONLY handles playing audio, freeing the network loop."""
        while True:
            audio_data = self.playback_queue.get()
            if audio_data is None:
                break
            if not self.stop_event.is_set():
                self._route_audio(audio_data)
            self.playback_queue.task_done()

    def _safe_write(self, stream, data):
        try:
            stream.write(data)
        except Exception:
            pass

    def _route_audio(self, audio_data):
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
            
        stereo_data = np.concatenate((audio_data, audio_data), axis=1)
        stereo_data_int16 = np.clip(stereo_data * 32767, -32768, 32767).astype(np.int16)

        futures = []
        if getattr(self, 'vts_stream', None) is not None:
            futures.append(self.executor.submit(self._safe_write, self.vts_stream, stereo_data_int16))
            
        if futures:
            concurrent.futures.wait(futures)

    def stop_with_fade(self, audio_queue):
        print("[Voice] 🛑 Interruption: Clearing hardware buffers...")
        self.stop_event.set()
        
        # 1. Properly drain the incoming sentence queue
        while True:
            try:
                audio_queue.get_nowait()
                audio_queue.task_done()
            except queue.Empty:
                break
                
        # 2. Properly drain the physical playback queue
        while True:
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
            except queue.Empty:
                break
        with audio_queue.mutex:
            audio_queue.queue.clear()
            
        # Clear the physical playback queue instantly
        with self.playback_queue.mutex:
            self.playback_queue.queue.clear()

    def reset_stop(self):
        self.stop_event.clear()

    def generate_voice_chunk(self, text: str, speed: float = 1.0, quality_steps: int = 8):
        clean_text = text.strip()
        if not clean_text:
            return None, self.sample_rate, 0.0
        
        # --- BRANCH A: LOCAL NATIVE KOKORO STREAM ENGINE ---
        if self.active_engine == "KOKORO":
            url = f"http://127.0.0.1:8008/v1/audio/speech"
            payload = {"input": clean_text, "voice": config.KOKORO_VOICE, "speed": config.KOKORO_SPEED}
            
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

        # --- BRANCH B: DOCKER CONTAINERIZED SUPERTONIC ENGINE ---
        else:
            clean_text = text.replace(",,", ", ").strip()
            url = f"http://{config.TTS_HOST}:{config.TTS_PORT}/v1/audio/speech"
            
            payload = {
                "input": clean_text,
                "voice": getattr(config, "TTS_VOICE", "F1"),
                "speed": float(speed),
                "total_steps": int(quality_steps)  # Matches the updated BaseModel
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
                print(f"[Voice Engine Error] Supertonic container stream link failure: {e}")
                return None, self.sample_rate, 0.0

    def __del__(self):
        try:
            if hasattr(self, 'playback_queue'):
                self.playback_queue.put(None)
            if hasattr(self, 'vts_stream') and self.vts_stream is not None:
                self.vts_stream.stop()
                self.vts_stream.close()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass