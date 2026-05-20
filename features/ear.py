import os
import time
import threading
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor
from collections import deque
import config

class Ear:
    def __init__(self):
        print(f"Loading Whisper Model '{config.WHISPER_MODEL}' on CUDA...")
        self.model = WhisperModel(config.WHISPER_MODEL, device="cuda", compute_type="int8_float16")
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
        
        # 2. THE CUTOFF GATE: Pipecat Smart Turn V3 (ONNX GPU)
        vad_filename = getattr(config, "VAD_MODEL_FILE", "smart-turn-v3.2-gpu.onnx")
        hf_vad_repo = getattr(config, "HF_VAD_REPO", "pipecat-ai/smart-turn-v3")
        vad_model_path = os.path.join(config.LOCAL_MODELS_DIR, vad_filename)

        if not os.path.exists(vad_model_path):
            print(f"\n[Ear] Local VAD model '{vad_filename}' not found in models directory.")
            print(f"[Ear] Initiating secure pull from HuggingFace Hub ({hf_vad_repo})...")
            try:
                hf_hub_download(
                    repo_id=hf_vad_repo,
                    filename=vad_filename,
                    local_dir=config.LOCAL_MODELS_DIR,
                    local_dir_use_symlinks=False
                )
                print("[Ear] ✅ VAD model successfully acquired.\n")
            except Exception as e:
                print(f"[Ear] FATAL: Failed to download VAD model: {e}")
                os._exit(1)

        print(f"[Ear] Loading Semantic Turn VAD (Cutoff Gate) via ONNX...")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.turn_session = ort.InferenceSession(
            vad_model_path, 
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.turn_input_name = self.turn_session.get_inputs()[0].name
        
        # Load the Whisper Feature Extractor to generate Mel Spectrograms
        self.turn_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        
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
            "last_pipecat_poll_time": 0.0 
        }
        
        consecutive_silence_samples = 0
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
                            
                else:
                    audio_buffer.extend(audio_chunk_flattened)
                    
                    if speech_prob < self.speech_end_threshold:
                        consecutive_silence_samples += len(audio_chunk_flattened)
                        silence_duration = consecutive_silence_samples / self.sample_rate
                        
                        current_time = time.perf_counter()
                        if silence_duration >= 0.3 and (current_time - state["last_pipecat_poll_time"] >= 0.15):
                            state["last_pipecat_poll_time"] = current_time
                            print(f"\n[VAD] ⏳ {int(silence_duration * 1000)}ms acoustic pause. Asking Smart Turn...")
                            
                            audio_snapshot = np.array(audio_buffer, dtype=np.float32).flatten()
                            
                            max_samples = 8 * self.sample_rate 
                            if len(audio_snapshot) > max_samples:
                                audio_snapshot = audio_snapshot[-max_samples:]
                            elif len(audio_snapshot) < max_samples:
                                pad_len = max_samples - len(audio_snapshot)
                                audio_snapshot = np.pad(audio_snapshot, (pad_len, 0), mode='constant')
                                
                            # Convert raw audio into a Log-Mel Spectrogram
                            mel_features = self.turn_processor(
                                audio_snapshot, 
                                sampling_rate=self.sample_rate, 
                                return_tensors="np"
                            ).input_features
                            
                            # Whisper pads to 3000 frames by default. V3 strictly expects 800 frames.
                            mel_spectrogram = mel_features[:, :, :800]
                            
                            ort_inputs = {
                                self.turn_input_name: mel_spectrogram
                            }
                            
                            logits = self.turn_session.run(None, ort_inputs)[0]
                            complete_prob = 1.0 / (1.0 + np.exp(-float(np.squeeze(logits))))
                            
                            if complete_prob >= self.turn_threshold:
                                print(f"  └─ Turn Complete (Conf: {complete_prob:.2f})")
                                break
                            else:
                                print(f"  └─ Continuing thought (Conf: {complete_prob:.2f}). Waiting...")
                        
                        if silence_duration > self.hard_fallback_timeout:
                            print(f"[VAD] Hard Fallback Triggered ({self.hard_fallback_timeout}s of dead air)")
                            break
                            
                    else:
                        consecutive_silence_samples = 0
                        with lock:
                            state["last_speech_frame_idx"] = len(audio_buffer)
                            state["physical_silence_start_time"] = time.perf_counter()
        
        start_compute_t = time.perf_counter()
        with lock:
            state["recording_active"] = False
            
        worker_thread.join(timeout=0.3)
        
        with lock:
            final_text = current_transcription[0]
            silence_start_ts = state["physical_silence_start_time"]
            
        whisper_time = time.perf_counter() - start_compute_t
        return final_text, whisper_time, silence_start_ts