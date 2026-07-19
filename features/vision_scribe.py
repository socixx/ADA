import time
import threading
import win32gui
import win32process
import psutil
import uiautomation as auto
from pynput import mouse
import os
import tempfile

# --- UPDATED NATIVE SDK NAMESPACE IMPORTS ---
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.graphics.imaging import BitmapDecoder
from winrt.windows.storage.streams import InMemoryRandomAccessStream
# ---------------------------------------------
import asyncio
import mss
from PIL import Image
import io

class VisionScribe:
    def __init__(self):
        print("[Vision Scribe] Initializing Zero-Latency Sensory Fusion Architecture...")
        self.live_workspace_state = {
            "active_app": "Unknown",
            "window_title": "Unknown",
            "highlighted_text": "[None]",
            "attention_ocr": "[None]",
            "desktop_audio": "[Initializing...]",
            "visual_scene": "[Waking up VLM...]"
        }
        self.stop_event = threading.Event()
        
        self.mouse_x, self.mouse_y = 0, 0
        self.sct = mss.mss()
        
        self.mouse_listener = mouse.Listener(on_move=self._on_mouse_move)
        self.mouse_listener.start()
        
        # UI & Visual Ticker Thread
        self.worker_thread = threading.Thread(target=self._scribe_loop, daemon=True)
        self.worker_thread.start()

        # --- NEW: Desktop Audio Loopback Thread ---
        self.audio_thread = threading.Thread(target=self._desktop_audio_worker, daemon=True)
        self.audio_thread.start()

        # --- NEW: Moondream2 Visual VLM Thread ---
        self.vlm_thread = threading.Thread(target=self._moondream_worker, daemon=True)
        self.vlm_thread.start()

    def _moondream_worker(self):
        """Pillar 5: The Visual 'Vibe Checker' (Moondream2)"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
            from PIL import Image
            import mss
            
            # Load the lightweight 1.8B model into fp16 on the GPU
            model_id = "vikhyatk/moondream2"
            revision = "2024-08-26"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            
            # --- THE ULTIMATE INTERCEPTION ---
            # 1. Load the configuration object separately FIRST
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
            
            # 2. Hard-patch the missing token into the config BEFORE the model ever sees it
            config.pad_token_id = tokenizer.eos_token_id
            
            # 3. Boot the model using our pre-patched config!
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                config=config, 
                trust_remote_code=True, 
                revision=revision, 
                torch_dtype=torch.float16, 
                device_map={"": "cuda"}
            )
            # ---------------------------------
            
            with mss.mss() as sct:
                while not self.stop_event.is_set():
                    try:
                        # Grab the primary monitor
                        monitor = sct.monitors[1]
                        sct_img = sct.grab(monitor)
                        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                        
                        # Generate the broad scene caption
                        enc_image = model.encode_image(img)
                        caption = model.answer_question(enc_image, "Describe this scene accurately and concisely.", tokenizer)
                        
                        self.live_workspace_state["visual_scene"] = caption
                    except Exception:
                        pass
                        
                    time.sleep(3) # Throttle to 1 frame every 3 seconds to save compute
                    
        except Exception as e:
            print(f"[Vision Scribe] Moondream2 failed to load: {e}")
            self.live_workspace_state["visual_scene"] = "[VLM Engine Offline]"

    def _on_mouse_move(self, x, y):
        self.mouse_x = x
        self.mouse_y = y

    def _get_active_window_info(self):
        """Pillar 1: Native OS Telemetry."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd: return "Unknown", "Unknown"
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                process = psutil.Process(pid)
                app_name = process.name()
            except Exception:
                app_name = "Unknown"
            return app_name, title
        except Exception:
            return "Unknown", "Unknown"

    def _get_uia_text(self):
        """Pillar 2: UIAutomation Scraper."""
        try:
            element = auto.ControlFromCursor()
            if element and element.Name:
                clean_name = str(element.Name).strip().replace('\n', ' ')
                return clean_name[:250] if len(clean_name) > 0 else "[None]"
        except Exception:
            pass
        return "[None]"

    def _get_micro_ocr(self):
        """Pillar 3: Snaps a bounding box and caches it to disk to bypass COM memory bugs."""
        try:
            import os
            import tempfile
            
            width, height = 300, 150
            left = self.mouse_x - (width // 2)
            top = self.mouse_y - (height // 2)
            
            monitor = {"top": top, "left": left, "width": width, "height": height}
            sct_img = self.sct.grab(monitor)
            
            # Convert to standard image format
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Write to a tiny, constantly overwritten temp file (cached in RAM by OS)
            temp_path = os.path.abspath(os.path.join(tempfile.gettempdir(), "ada_vision_buffer.png"))
            img.save(temp_path, format="PNG")
            
            return asyncio.run(self._run_winrt_ocr(temp_path))
        except Exception:
            return "[None]"
        
    def _desktop_audio_worker(self):
        """Pillar 4: Intercepts Windows Desktop Audio via pure WASAPI buffer."""
        try:
            import pyaudiowpatch as pyaudio
            import speech_recognition as sr
            import wave
            import io

            r = sr.Recognizer()

            with pyaudio.PyAudio() as p:
                # 1. Locate the active default speakers
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                
                # 2. Find the hidden WASAPI loopback channel for those exact speakers
                loopback_device = None
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        loopback_device = loopback
                        break
                        
                if not loopback_device:
                    self.live_workspace_state["desktop_audio"] = "[Loopback Device Not Found]"
                    return

                # 3. Configure strict WASAPI hardware parameters
                CHUNK = 4096
                FORMAT = pyaudio.paInt16
                CHANNELS = loopback_device["maxInputChannels"]
                RATE = int(loopback_device["defaultSampleRate"])
                RECORD_SECONDS = 4  # 4-second rolling transcription bursts

                while not self.stop_event.is_set():
                    try:
                        # Open the raw byte stream
                        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                                        input=True, frames_per_buffer=CHUNK, 
                                        input_device_index=loopback_device["index"])

                        frames = []
                        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                            data = stream.read(CHUNK, exception_on_overflow=False)
                            frames.append(data)

                        stream.stop_stream()
                        stream.close()

                        # Convert raw frames to an in-memory WAV file instantly
                        wav_io = io.BytesIO()
                        with wave.open(wav_io, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(frames))
                        
                        wav_io.seek(0)

                        # Feed the clean memory file directly into the recognizer engine
                        with sr.AudioFile(wav_io) as source:
                            audio = r.record(source)

                        text = r.recognize_google(audio)
                        self.live_workspace_state["desktop_audio"] = text
                        
                    except sr.UnknownValueError:
                        self.live_workspace_state["desktop_audio"] = "[Unintelligible / Music]"
                    except Exception:
                        pass # Ignore transient drops between audio tracks
                        
        except Exception as e:
            print(f"[Vision Scribe] Audio loopback failed: {e}")
            self.live_workspace_state["desktop_audio"] = "[Driver Error - Check Interpreter]"

    async def _run_winrt_ocr(self, file_path):
        """Let Windows natively read the file, completely bypassing Python's broken IBuffer conversion."""
        try:
            from winrt.windows.media.ocr import OcrEngine
            from winrt.windows.graphics.imaging import BitmapDecoder
            from winrt.windows.storage import StorageFile
            
            # 1. Native Windows File Read (0 = FileAccessMode.Read)
            file = await StorageFile.get_file_from_path_async(file_path)
            stream = await file.open_async(0) 
            
            # 2. Decode the image natively
            decoder = await BitmapDecoder.create_async(stream)
            software_bitmap = await decoder.get_software_bitmap_async()
            
            # 3. Run native OCR
            engine = OcrEngine.try_create_from_user_profile_languages()
            if not engine: 
                return "[None]"
            
            result = await engine.recognize_async(software_bitmap)
            return result.text.strip() if result.text else "[None]"
            
        except Exception as e:
            print(f"[Vision Scribe OCR Engine Error]: {e}")
            return "[None]"

    def _scribe_loop(self):
        """Runs continuously in the background, updating state variables at 2 FPS."""
        while not self.stop_event.is_set():
            try:
                app_name, title = self._get_active_window_info()
                self.live_workspace_state["active_app"] = app_name
                self.live_workspace_state["window_title"] = title
                
                # UIAutomation Fallback chain
                self.live_workspace_state["highlighted_text"] = self._get_uia_text()
                
                # Snaps targeted bounding box under mouse cursor using Win11 engine
                self.live_workspace_state["attention_ocr"] = self._get_micro_ocr()
                
            except Exception:
                pass
            time.sleep(0.5)

    def get_ticker_text(self):
        """Compiles the rolling state variables into an explicit, context-aware LLM prompt."""
        state = self.live_workspace_state
        
        ticker = (
            "\n[SYSTEM DIRECTIVE: LIVE SENSORY WORKSPACE STATE]\n"
            "This is raw real-time data scraped from the user's PC screen and speakers. "
            "CRITICAL RULES: \n"
            "1. 'Desktop Audio' is background media (YouTube, Games, Spotify). It is NEVER the user speaking to you.\n"
            "2. Do NOT assume you made, created, or own any of this content.\n"
            "3. Only reference this data if it contextually answers the user's prompt (e.g. 'what is this song?', 'what am i watching?').\n"
            f"- Active Window: {state['active_app']} (\"{state['window_title']}\")\n"
            f"- UI Element under mouse: {state['highlighted_text']}\n"
            f"- Visual OCR around mouse: {state.get('attention_ocr', '[None]')}\n"
            f"- Desktop Audio Transcript: {state.get('desktop_audio', '[None]')}\n"
            f"- Visual Scene: {state.get('visual_scene', '[None]')}\n" # <-- NEW
            "[/END SYSTEM DIRECTIVE]\n"
        )
        return ticker