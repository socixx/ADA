import os
import time
import numpy as np
import sounddevice as sd
import torch
import io
import soundfile as sf
import requests
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor
from collections import deque
import config

class Ear:
    def __init__(self):
        self.audio_port = getattr(config, "AUDIO_PORT", 8008)
        print(f"[Ear] Binding to Audio API Engine on Port {self.audio_port}...")
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
        
        self.turn_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        
        self.speech_start_threshold = 0.55  
        self.speech_end_threshold = 0.45    
        self.turn_threshold = config.SEMANTIC_TURN_THRESHOLD
        self.hard_fallback_timeout = config.SILENCE_TIMEOUT 
        self.volume_threshold = config.VAD_THRESHOLD

    def listen_and_transcribe(self):
        print("\n[Ada is listening...]")
        audio_buffer = []
        pre_roll_buffer = deque(maxlen=int(self.sample_rate * 0.5))
        
        state = {
            "speech_started": False, 
            "physical_silence_start_time": 0.0,
            "last_pipecat_poll_time": 0.0 
        }
        
        consecutive_silence_samples = 0
        
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
                        state["speech_started"] = True
                        print("\n[VAD] 🎙️ Speech Triggered (Silero)")
                        
                        audio_buffer.extend(pre_roll_buffer)
                        audio_buffer.extend(audio_chunk_flattened)
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
                                
                            mel_features = self.turn_processor(
                                audio_snapshot, 
                                sampling_rate=self.sample_rate, 
                                return_tensors="np"
                            ).input_features
                            
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
                        state["physical_silence_start_time"] = time.perf_counter()
        
        # --- TRANSCRIBE THE COMPLETED TURN ---
        start_compute_t = time.perf_counter()
        final_text = ""
        
        audio_snapshot = np.array(audio_buffer, dtype=np.float32).flatten()
        
        try:
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_snapshot, self.sample_rate, format='WAV', subtype='PCM_16')
            wav_io.seek(0)
            
            url = f"http://localhost:{self.audio_port}/v1/audio/transcriptions"
            files = {'file': ('audio.wav', wav_io, 'audio/wav')}
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                final_text = response.json().get("text", "").strip()
            else:
                print(f"[Ear] API Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[Ear] Transcription Connection Error: {e}")
            
        whisper_time = time.perf_counter() - start_compute_t
        silence_start_ts = state["physical_silence_start_time"]
        
        return final_text, whisper_time, silence_start_ts