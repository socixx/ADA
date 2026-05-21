import time
import threading
import sounddevice as sd
import numpy as np
import config
import concurrent.futures
import os
import requests

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
        self.engine = config.ACTIVE_TTS
        self.audio_port = getattr(config, "AUDIO_PORT", 8008)
        print(f"[Voice] Binding to Audio API Engine on Port {self.audio_port}...")

        # Persistent connection
        self.http_session = requests.Session() 

        self.stop_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.sample_rate = 24000
        self.voice_name = getattr(config, "KOKORO_VOICE", "af_bella")

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
        
        url = f"http://127.0.0.1:{self.audio_port}/v1/audio/speech"
        payload = {
            "input": clean_text,
            "voice": self.voice_name,
            "speed": getattr(config, "KOKORO_SPEED", 1.1)
        }

        try:
            # Blast request - fetches the entire audio array instantly
            response = self.http_session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            ttfa = time.perf_counter() - start_t
            
            # Convert raw bytes back to numpy array
            audio_data = np.frombuffer(response.content, dtype=np.float32)
            if len(audio_data) == 0:
                return None, self.sample_rate, ttfa
                
            audio_data = audio_data.reshape(-1, 1)

            # EXACT REPLICA OF LOCAL PIPELINE: Slice into 0.1s chunks and buffer to sounddevice
            block_size = int(self.sample_rate * 0.1) 
            
            for i in range(0, len(audio_data), block_size):
                if self.stop_event.is_set():
                    # Trail-off logic for interruptions
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
                
                # Normal 0.1s playback block
                end_idx = min(i + block_size, len(audio_data))
                self._dual_write(audio_data[i : end_idx])

            return None, self.sample_rate, ttfa
            
        except Exception as e:
            print(f"[Voice] Stream Error: {e}")
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