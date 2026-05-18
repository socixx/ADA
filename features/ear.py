import time
import threading
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from transformers import Wav2Vec2Processor
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import _Wav2Vec2ForEndpointing
import config
from collections import deque

# ==============================================================================
# --- HOTFIX FOR TRANSFORMERS 4.40+ COMPATIBILITY ---
orig_init = _Wav2Vec2ForEndpointing.__init__
def patched_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    if not hasattr(self, 'all_tied_weights_keys'):
        self.all_tied_weights_keys = {}  
_Wav2Vec2ForEndpointing.__init__ = patched_init
# ==============================================================================

class Ear:
    def __init__(self):
        print(f"Loading Whisper Model '{config.WHISPER_MODEL}' on CUDA...")
        self.model = WhisperModel(config.WHISPER_MODEL, device="cuda", compute_type="float16")
        self.sample_rate = 16000
        self.channels = 1
        
        # 1. THE WAKE GATE: Silero
        print("[Ear] Loading Neural Silero VAD (Wake Gate)...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.vad_model = self.vad_model.to("cuda")
        
        # 2. THE CUTOFF GATE: Pipecat Smart Turn V2
        print(f"[Ear] Loading Semantic Turn VAD '{config.SEMANTIC_VAD_MODEL}' (Cutoff Gate)...")
        self.turn_processor = Wav2Vec2Processor.from_pretrained(config.SEMANTIC_VAD_MODEL)
        self.turn_classifier = _Wav2Vec2ForEndpointing.from_pretrained(config.SEMANTIC_VAD_MODEL).to("cuda")
        self.turn_classifier.eval() 
        
        self.speech_start_threshold = 0.55  
        self.speech_end_threshold = 0.45    
        self.turn_threshold = config.SEMANTIC_TURN_THRESHOLD
        self.hard_fallback_timeout = config.SILENCE_TIMEOUT 
        self.volume_threshold = config.VAD_THRESHOLD

    def listen_and_transcribe(self):
        print("\n[Ada is listening...]")
        audio_buffer = []
        current_transcription = [""]
        pre_roll_buffer = deque(maxlen=int(self.sample_rate * 0.5))
        
        state = {
            "speech_started": False, 
            "recording_active": True,
            "last_speech_frame_idx": 0,
            "physical_silence_start_time": 0.0,
            "last_pipecat_poll_time": 0.0 # Replaces the lockout flag
        }
        
        consecutive_silence_samples = 0
        hard_silence_counter = 0
        lock = threading.Lock()
        
        def bg_transcribe_worker():
            last_processed_idx = 0
            while True:
                for _ in range(5):
                    if not state["recording_active"]:
                        break
                    time.sleep(0.04)
                
                with lock:
                    is_active = state["recording_active"]
                    target_idx = state["last_speech_frame_idx"]
                    speech_begun = state["speech_started"]
                
                if not speech_begun or target_idx <= last_processed_idx:
                    if not is_active:
                        break
                    continue
                
                with lock:
                    audio_snapshot = np.array(audio_buffer[:target_idx], dtype=np.float32).flatten()
                    last_processed_idx = target_idx
                
                try:
                    segments, _ = self.model.transcribe(
                        audio_snapshot, 
                        beam_size=1,
                        language="en",
                        condition_on_previous_text=False,
                        temperature=0.0,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=250)
                    )
                    text = "".join([segment.text for segment in segments]).strip()
                    with lock:
                        current_transcription[0] = text
                except Exception:
                    pass
                    
                if not is_active:
                    break

        worker_thread = threading.Thread(target=bg_transcribe_worker, daemon=True)
        worker_thread.start()

        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype='float32', blocksize=512) as stream:
            while True:
                audio_chunk, _ = stream.read(512)
                audio_chunk_flattened = audio_chunk.flatten()
                
                tensor_chunk = torch.from_numpy(audio_chunk_flattened).to("cuda")
                with torch.inference_mode():
                    speech_prob = self.vad_model(tensor_chunk, self.sample_rate).item()
                
                # --- PHASE 1: WAITING FOR SPEECH ---
                if not state["speech_started"]:
                    pre_roll_buffer.extend(audio_chunk_flattened)
                    
                    if speech_prob > self.speech_start_threshold:
                        with lock:
                            state["speech_started"] = True
                            print("\n[VAD] 🎙️ Speech Triggered (Silero)")
                            
                            audio_buffer.extend(pre_roll_buffer)
                            audio_buffer.extend(audio_chunk_flattened)
                            
                            state["last_speech_frame_idx"] = len(audio_buffer)
                            state["physical_silence_start_time"] = time.perf_counter()
                            
                # --- PHASE 2: ACTIVE LISTENING ---
                else:
                    audio_buffer.extend(audio_chunk_flattened)
                    
                    # IF THE USER STOPS MAKING SOUND (Silero goes below 0.45)
                    if speech_prob < self.speech_end_threshold:
                        consecutive_silence_samples += len(audio_chunk_flattened)
                        silence_duration = consecutive_silence_samples / self.sample_rate
                        
                        # CONTINUOUS POLLING: Check Pipecat every 150ms after the initial 300ms pause
                        current_time = time.perf_counter()
                        if silence_duration >= 0.3 and (current_time - state["last_pipecat_poll_time"] >= 0.15):
                            state["last_pipecat_poll_time"] = current_time
                            print(f"\n[VAD] ⏳ {int(silence_duration * 1000)}ms acoustic pause. Asking Pipecat...")
                            
                            audio_snapshot = np.array(audio_buffer, dtype=np.float32).flatten()
                            if len(audio_snapshot) > 8 * self.sample_rate:
                                audio_snapshot = audio_snapshot[-8 * self.sample_rate:]
                                
                            inputs = self.turn_processor(
                                audio_snapshot, 
                                sampling_rate=self.sample_rate, 
                                return_tensors="pt",
                                return_attention_mask=True  
                            )
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                            
                            with torch.inference_mode():
                                outputs = self.turn_classifier(**inputs)
                                if isinstance(outputs, dict):
                                    logits = outputs["logits"]
                                elif hasattr(outputs, "logits"):
                                    logits = outputs.logits
                                else:
                                    logits = outputs[0]
                                    
                                complete_prob = torch.sigmoid(logits).squeeze().item()
                            
                            if complete_prob >= self.turn_threshold:
                                print(f"  └─ ⚡ Turn Complete (Conf: {complete_prob:.2f})")
                                break
                            else:
                                print(f"  └─ 🤔 Continuing thought (Conf: {complete_prob:.2f}). Waiting...")
                        
                        # --- HARD FALLBACK ---
                        if silence_duration > self.hard_fallback_timeout:
                            print(f"[VAD] 🛑 Hard Fallback Triggered ({self.hard_fallback_timeout}s of dead air)")
                            break
                            
                    # IF THE USER IS ACTIVELY SPEAKING
                    else:
                        consecutive_silence_samples = 0
                        
                        with lock:
                            state["last_speech_frame_idx"] = len(audio_buffer)
                            state["physical_silence_start_time"] = time.perf_counter()
        
        # --- CLEANUP & RETURN ---
        start_compute_t = time.perf_counter()
        with lock:
            state["recording_active"] = False
            
        worker_thread.join(timeout=0.3)
        
        with lock:
            final_text = current_transcription[0]
            silence_start_ts = state["physical_silence_start_time"]
            
        whisper_time = time.perf_counter() - start_compute_t
        return final_text, whisper_time, silence_start_ts
    