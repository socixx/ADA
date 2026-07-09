import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import queue
import threading
import re
import random
import subprocess
import requests
from loguru import logger
import eel  
import json
import glob
import urllib.request
import signal

# Scrub logger bloat
logger.remove()
logger.add(sys.stderr, level="WARNING") 

import config
from features.brain import Brain
from features.voice import Voice
from features.ear import Ear
from features.vts_bridge import VTSBridge
from features.eye import Eye

# --- GLOBAL QUEUES & STATE ---
audio_queue = queue.Queue()
input_queue = queue.Queue()
telemetry_data = {"first_audio_timestamp": None}
ada_state = {"is_speaking": False, "last_generation": "", "spoken_text": ""}

# --- GLOBAL NODE REFERENCES (For UI Hooks) ---
brain_node = None
voice_node = None
vts_node = None
eye_node = None
ear_node = None
active_llm_thread = None

# ==========================================
# EEL UI EXPOSED FUNCTIONS (Bridge to JS)
# ==========================================
@eel.expose
def ui_interrupt_ada():
    brain_node.abort_event.set()
    voice_node.stop_with_fade(audio_queue)
    try: eel.update_telemetry_detailed("llm", "INTERRUPTED", "#ef4444", "Generation Halted") 
    except Exception: pass

@eel.expose
def ui_trigger_vision():
    global eye_node
    print("[System] UI Vision payload requested.")
    try:
        eel.update_vision_card("Scanning workspace...")
        vision_result = eye_node.look_at_screen()
        eel.update_vision_card(vision_result) 
        eel.update_chat("system", f"Vision Context Synchronized Engine Side.", True) 
    except Exception as e:
        print(f"[UI Link Error] Vision bridge failure: {e}")

@eel.expose
def ui_update_temp_config(key, value):
    # Applies sliders instantly to current runtime memory
    setattr(config, key, value)
    print(f"[Config] Dynamic shift applied for current session: {key} = {value}")

@eel.expose
def ui_save_config_to_disk(settings_dict):
    # Permanently saves settings to config.py using Regex to protect your formatting
    print(f"[Config] Writing new defaults to config.py...")
    try:
        with open("config.py", "r", encoding="utf-8") as f:
            content = f.read()

        for key, val in settings_dict.items():
            setattr(config, key, val) 
            if isinstance(val, str):
                content = re.sub(rf"^{key}\s*=\s*.*$", f'{key} = "{val}"', content, flags=re.MULTILINE)
            else:
                content = re.sub(rf"^{key}\s*=\s*.*$", f'{key} = {val}', content, flags=re.MULTILINE)
        
        with open("config.py", "w", encoding="utf-8") as f:
            f.write(content)
            
        print("[Config] ✅ Save successful. Changes are now permanent.")
        return True
    except Exception as e:
        print(f"[Config] 🛑 Save failed: {e}")
        return False
    
# ==========================================
# EEL UI DATA FETCHERS (Memory, Logs, VTS)
# ==========================================

@eel.expose
def ui_fetch_memory():
    """Reads live context and pulls ALL .json files from the memory/ folder."""
    # 1. Short Term Active Context
    short_term_text = "Brain node offline."
    if brain_node and hasattr(brain_node, 'history'):
        lines = []
        for msg in brain_node.history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        short_term_text = "\n\n".join(lines)
    
    # 2. Long Term Disk Readers (Scans the memory/ folder)
    mem_dir = "memory"
    long_term_data = {}
    
    if os.path.exists(mem_dir) and os.path.isdir(mem_dir):
        for file_path in glob.glob(os.path.join(mem_dir, "*.json")):
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    long_term_data[filename] = json.load(f)
            except Exception as e:
                long_term_data[filename] = f"Error reading file: {e}"
                
        if long_term_data:
            long_term_text = json.dumps(long_term_data, indent=2)
        else:
            long_term_text = "No JSON files found in the 'memory/' directory."
    else:
        long_term_text = "Memory directory 'memory/' not found on disk."

    return {"short_term": short_term_text, "long_term": long_term_text}

@eel.expose
def ui_fetch_logs():
    """Reads every historical conversation session from the JSONL dataset for the UI browser."""
    log_path = "training_logs.jsonl"
    if not os.path.exists(log_path):
        return '{"messages": [{"role": "system", "content": "No conversation history found on disk yet. Start talking to Ada to generate live transcripts!"}]}'
        
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            # Reads the entire file, keeping every historical turn intact
            lines = f.readlines()
            if not lines:
                return '{"messages": [{"role": "system", "content": "Transcript dataset is currently empty."}]}'
            
            # Combine all lines back together to pass over the WebSocket bridge
            return "".join(lines)
    except Exception as e:
        return f'{{"messages": [{{"role": "system", "content": "Error reading training dataset: {e}"}}]}}'


@eel.expose
def ui_fetch_vts_state():
    """Polls the VTube Studio Bridge for a live connection state."""
    global vts_node
    state = {
        "connected": False,
        "model_name": "Bridge Disconnected"
    }
    
    if vts_node:
        try:
            # FIX: Changed from 'is_connected' to 'connected' to match vts_bridge.py
            is_live = getattr(vts_node, 'connected', False)
            state["connected"] = is_live
            state["model_name"] = "Ada_Avatar" if is_live else "Unauthorized / No Link"
        except Exception as e:
            print(f"[UI Link Error] VTS polling failure: {e}")
            state["model_name"] = "Connection Error"
            
    return state

@eel.expose
def ui_get_current_settings():
    """Reads the live configuration parameters from config.py on runtime startup."""
    return {
        "LLM_MIN_P": getattr(config, "LLM_MIN_P", 0.05),
        "LLM_TOP_P": getattr(config, "LLM_TOP_P", 0.9),
        "LLM_MAX_TOKENS": getattr(config, "LLM_MAX_TOKENS", 4096),
        "TTS_SPEED": getattr(config, "TTS_SPEED", 1.15),
        "TTS_QUALITY_STEPS": getattr(config, "TTS_QUALITY_STEPS", 5)
    }

@eel.expose
def ui_trigger_vts(action_name):
    """Triggers specific VTS animations manually from the UI."""
    if vts_node:
        print(f"[UI] Forcing VTS expression: {action_name}")
        # Add asterisks to match how your LLM worker parses it
        vts_node.trigger_action(f"*{action_name}*")


@eel.expose
def ui_force_memory_prune():
    """Allows the user to manually trigger the deep-sleep summarization process."""
    if brain_node:
        print("[UI] Manual Memory Consolidation Triggered.")
        try: eel.update_chat("system", "Manual memory consolidation initiated...")()
        except: pass
        
        # Runs the deep sleep function in a background thread so it doesn't freeze the UI
        threading.Thread(target=brain_node.memory.deep_sleep_consolidation, daemon=True).start()

# ==========================================
# CONTAINER INFRASTRUCTURE
# ==========================================
def shutdown_containers():
    containers = [
        getattr(config, "LLM_CONTAINER_NAME", None),
        getattr(config, "VISION_CONTAINER_NAME", None),
        getattr(config, "TTS_CONTAINER_NAME", None),
    ]
    target_containers = [c for c in containers if c]
    if target_containers:
        print(f"\n[System] Terminating nodes: {target_containers}...")
        subprocess.Popen(
            ["docker", "stop", "-t", "1"] + target_containers, 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

def ensure_local_models():
    llm_path = os.path.join(config.LOCAL_MODELS_DIR, config.LLM_MODEL)
    if os.path.exists(llm_path):
        print(f"[System] Verified local model matrix: {config.LLM_MODEL}")
    else:
        print(f"[System] ⚠️ LLM Matrix file missing at: {llm_path}")

def get_container_status(container_name):
    try:
        res = subprocess.run(["docker", "ps", "-a", "--format", "{{.Status}}", "-f", f"name={container_name}"], capture_output=True, text=True)
        return res.stdout.strip()
    except Exception: return ""

def wait_for_node(url, name, timeout=300):
    start = time.time()
    print(f"[System] Waiting for {name} matrix validation...")
    while time.time() - start < float(timeout):
        try:
            res = requests.get(url, timeout=2)
            if res.status_code == 200: return True
        except requests.exceptions.RequestException:
            time.sleep(2)
    print(f"\n[System] 🛑 FATAL: {name} container crashed.")
    exit(1)

def ensure_vllm_containers():
    print("[System] Verifying local container runtime states...")
    status_llama = get_container_status(config.LLM_CONTAINER_NAME)
    if not status_llama:
        print(f"[System] Launching primary text architecture node: {config.LLM_CONTAINER_NAME}...")
        abs_checkpoints_dir = os.path.abspath(config.LOCAL_MODELS_DIR)
        cmd = [
            "docker", "run", "-d", "--name", config.LLM_CONTAINER_NAME, "--gpus", "all", "--ipc=host",
            "-p", f"{config.LLM_PORT}:8000",
            "-v", f"{config.HF_CACHE_DIR}:/root/.cache/huggingface",
            "-v", f"{abs_checkpoints_dir}:/app/checkpoints",
            "vllm/vllm-openai:latest", "--model", f"/app/checkpoints/{config.LLM_MODEL}",
            "--tokenizer", "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", "0.60", "--max-model-len", "4096"
        ]
        subprocess.run(cmd)
    elif "Up" not in status_llama:
        subprocess.run(["docker", "start", config.LLM_CONTAINER_NAME])
    wait_for_node(f"http://localhost:{config.LLM_PORT}/v1/models", "Text Engine (Llama)")

    if getattr(config, "ACTIVE_TTS", "TTS") == "TTS":
        status_tts = get_container_status(config.TTS_CONTAINER_NAME)
        target_model_dir = os.path.join(config.LOCAL_MODELS_DIR, "supertonic_3")
        marker_file = os.path.join(target_model_dir, "model.onnx")
        
        if not os.path.exists(marker_file):
            print(f"[System] Syncing Supertonic 3 weights from Hugging Face...")
            from huggingface_hub import snapshot_download
            os.makedirs(target_model_dir, exist_ok=True)
            snapshot_download(repo_id=config.TTS_MODEL, local_dir=target_model_dir, cache_dir=config.HF_CACHE_DIR)

        if not status_tts:
            print(f"[System] Building Local Audio Engine node: {config.TTS_CONTAINER_NAME}...")
            subprocess.run(["docker", "build", "-t", "supertonic-tts-node", "./supertonic_node"])
            cmd = [
                "docker", "run", "-d", "--name", config.TTS_CONTAINER_NAME, "--ipc=host",
                "-p", f"{config.TTS_PORT}:8000", "-v", f"{os.path.abspath(target_model_dir)}:/app/model",
                "supertonic-tts-node"
            ]
            subprocess.run(cmd)
        elif "Up" not in status_tts:
            subprocess.run(["docker", "start", config.TTS_CONTAINER_NAME])
        wait_for_node(f"http://{config.TTS_HOST}:{config.TTS_PORT}/health", "Audio Engine", timeout=300)

# ==========================================
# HARDWARE WORKERS
# ==========================================
def background_audio_worker(voice, q, telemetry, state):
    try: eel.update_telemetry_detailed("tts", "Idle", "#64748b", "")
    except: pass
    
    is_first_sentence_of_turn = True
    while True:
        item = q.get()
        if item is None: break
        if item == "__RESET_TURN__":
            is_first_sentence_of_turn = True
            state["spoken_text"] = ""
            voice.reset_stop()
            q.task_done()
            continue

        sentence, queue_entry_time = item
        state["is_speaking"] = True
        
        # Phase 1: Synthesizing the Audio Array
        try: eel.update_telemetry_detailed("tts", "Synthesizing...", "#cba6f7", f"{len(sentence)} chars") 
        except: pass

        # Assuming generate_voice_chunk returns (wav_data, duration, ttfa)
        # If your voice class blocks here until it finishes playing, the UI will reflect that!
        wav, duration, internal_ttfa = voice.generate_voice_chunk(sentence)
        state["spoken_text"] += sentence + " "

        # Phase 2: Playing the Audio
        # If generate_voice_chunk is non-blocking and plays async, we sleep for the duration
        try: eel.update_telemetry_detailed("tts", "Playing Audio...", "#a6e3a1", f"{(duration if duration else 0.0):.1f}s") 
        except: pass
        
        if is_first_sentence_of_turn:
            telemetry["first_audio_timestamp"] = queue_entry_time + (internal_ttfa if internal_ttfa else 0.0)
            is_first_sentence_of_turn = False

        state["is_speaking"] = False
        try: eel.update_telemetry_detailed("tts", "Idle", "#64748b", "") 
        except: pass
        
        q.task_done()

def background_ear_worker(ear, in_q):
    while True:
        # --- NEW TELEMETRY UPDATE ---
        try: eel.update_telemetry("ear", "Listening...", "#a6e3a1")
        except: pass
        
        user_text, whisper_time, silence_start_timestamp = ear.listen_and_transcribe()
        
        if user_text.strip():
            # --- NEW TELEMETRY UPDATE ---
            try: eel.update_telemetry("ear", "Processing VAD...", "#f9e2af")
            except: pass
            in_q.put((user_text, whisper_time, silence_start_timestamp))

def auto_log_turn_to_transcript(user_text, assistant_text):
    """Automatically logs a completed turn pair to the JSONL dataset in real-time."""
    if not user_text.strip() or not assistant_text.strip():
        return
        
    # Clean up any inner action asterisks from the transcript text if desired,
    # or leave them in so she learns to output VTS expressions!
    messages = [
        {"role": "user", "content": user_text.strip()},
        {"role": "assistant", "content": assistant_text.strip()}
    ]
    
    log_path = "training_logs.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"messages": messages}) + "\n")
        print(f"[Transcript System] Live-logged completed turn block to {log_path}")
    except Exception as e:
        print(f"[Transcript System] Live-log failed: {e}")

# ==========================================
# COGNITIVE LOOP 
# ==========================================
def cognitive_loop():
    global brain_node, voice_node, vts_node, eye_node, ear_node, active_llm_thread
    
    while True:
        try: eel.update_state("Listening...", "#a6e3a1") 
        except: pass

        user_text, whisper_time, silence_start_timestamp = input_queue.get()
        print(f"\n[You]: {user_text}")
        
        try: eel.update_chat("user", user_text) 
        except: pass

        if not audio_queue.empty() or ada_state["is_speaking"] or (active_llm_thread and active_llm_thread.is_alive()):
            print("[System] Interruption detected. Evaluating intent...")
            actual_spoken = ada_state["spoken_text"].strip()
            should_yield = brain_node.evaluate_interruption(actual_spoken, user_text)

            if should_yield:
                brain_node.abort_event.set()
                voice_node.stop_with_fade(audio_queue)
                if active_llm_thread and active_llm_thread.is_alive():
                    active_llm_thread.join(timeout=1.0)

                brain_node.history.append({"role": "assistant", "content": actual_spoken + f"— [INTERRUPTED]"})
                if any(word in user_text.lower() for word in ["stop", "wait", "nevermind", "cancel"]):
                    user_text = f"*[Interrupts Ada]* {user_text} (System Directive: Acknowledge interruption and drop topic.)"
                else:
                    user_text = f"*[Interrupts Ada]* {user_text}"
                time.sleep(0.6)
            else:
                continue

        clean_text = user_text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()
        
        if "good night ada" in clean_text:
            brain_node.abort_event.set()
            if active_llm_thread and active_llm_thread.is_alive(): active_llm_thread.join(timeout=1.0)
            voice_node.stop_with_fade(audio_queue)
            
            brain_node.abort_event.clear() 
            goodbye_msg = "Goodnight! Consolidating my memories now..."
            try: eel.update_chat("ada", goodbye_msg) 
            except: pass
            
            audio_queue.put("__RESET_TURN__")
            audio_queue.put((goodbye_msg, time.perf_counter()))
            audio_queue.join()

            # --- AUTOMATIC DATASET EXTRACTOR ---
            auto_log_session_to_transcript()
            # ------------------------------------

            brain_node.memory.deep_sleep_consolidation()
            shutdown_containers()
            os._exit(0)

        trigger_phrases = ["look at my screen", "read my screen", "what am i looking at", "can you see my screen"]
        screen_context = ""
        
        if any(phrase in clean_text for phrase in trigger_phrases):
            try: eel.update_state("Processing Vision...", "#89b4fa") 
            except: pass
            
            fillers = ["*leans in* Let's see here...", "*looks closely* Give me one sec..."]
            looking_msg = random.choice(fillers)
            
            audio_queue.put("__RESET_TURN__")
            audio_queue.put((looking_msg, time.perf_counter()))
            vts_node.trigger_action("looks at screen") 
            audio_queue.join() 
            
            vision_result = eye_node.look_at_screen()
            screen_context = f"\n[SYSTEM OBSERVATION: Ada just looked at the user's screen. She saw: {vision_result}]\n"
            
            try: eel.update_chat("system", f"Vision Context Loaded: {vision_result[:50]}...") 
            except: pass

        def llm_worker(text_to_process, visual_context, is_retry=False):
            try: eel.update_telemetry_detailed("llm", "Prompt Processing...", "#f9e2af", "") 
            except Exception: pass

            print("\n[ADA]: ", end="", flush=True)

            telemetry_data["first_audio_timestamp"] = None
            brain_node.abort_event.clear()
            
            if not is_retry: audio_queue.put("__RESET_TURN__")

            buffer = ""
            ada_state["last_generation"] = ""
            sentence_pool = []
            is_first_sentence = True
            target_window = getattr(config, "SENTENCE_WINDOW", 1)
            
            ABBREVIATIONS = r'\b(mr|ms|mrs|dr|st|inc|co|vs|approx|etc)\.$'

            # --- TPS TRACKING SETUP ---
            start_time = time.perf_counter()
            token_count = 0

            for chunk in brain_node.get_response_stream(text_to_process, screen_context=visual_context):
                if brain_node.abort_event.is_set(): break
                    
                # Update Tokens and TPS Live
                token_count += 1
                elapsed = time.perf_counter() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0.0
                
                try: eel.update_telemetry_detailed("llm", "Generating...", "#3b82f6", f"{token_count} tok | {tps:.1f} tps")
                except Exception: pass

                print(chunk, end="", flush=True)
                buffer += chunk
                ada_state["last_generation"] += chunk
                
                if any(p in chunk for p in ['.', '!', '?']):
                    if buffer.endswith('..') or buffer.endswith('...'):
                        continue
                        
                    idx = max(buffer.rfind('.'), buffer.rfind('!'), buffer.rfind('?'))
                    raw_sentence = buffer[: idx + 1]
                    
                    if re.search(ABBREVIATIONS, raw_sentence.strip().lower()):
                        continue
                        
                    buffer = buffer[idx + 1:]
                    clean_sentence = raw_sentence.strip()
                    
                    if clean_sentence:
                        actions = re.findall(r'\*(.*?)\*', clean_sentence)
                        for action in actions: vts_node.trigger_action(action.strip())
                        
                        speech_only = re.sub(r'\*.*?\*', '', clean_sentence).strip()
                        speech_only = speech_only.replace("'", "").replace('"', '').strip()
                        
                        if len(speech_only) > 1:
                            try: eel.update_chat("ada", speech_only, bool(is_first_sentence))
                            except Exception: pass

                            if is_first_sentence:
                                audio_queue.put((speech_only, time.perf_counter()))
                                is_first_sentence = False
                            else:
                                sentence_pool.append(speech_only)
                                if len(sentence_pool) >= target_window:
                                    combined_chunk = " ".join(sentence_pool)
                                    audio_queue.put((combined_chunk, time.perf_counter()))
                                    sentence_pool.clear()

            remaining_text = buffer.strip()
            if remaining_text and not brain_node.abort_event.is_set():
                speech_only = re.sub(r'\*.*?\*', '', remaining_text).strip()
                if speech_only:
                    try: eel.update_chat("ada", speech_only, bool(is_first_sentence))
                    except Exception: pass
                    sentence_pool.append(speech_only)

            if sentence_pool and not brain_node.abort_event.is_set():
                combined_chunk = " ".join(sentence_pool)
                audio_queue.put((combined_chunk, time.perf_counter()))
            
            print()
            try: eel.update_telemetry_detailed("llm", "Idle", "#64748b", f"Total: {token_count} tok") 
            except Exception: pass

            # --- LIVE CRASH-PROOF TRANSCRIPT SYSTEM ---
            # Logs the interaction immediately upon turn completion
            if not brain_node.abort_event.is_set():
                final_response = ada_state.get("last_generation", "").strip()
                auto_log_turn_to_transcript(text_to_process, final_response)
            # ------------------------------------------

        active_llm_thread = threading.Thread(target=llm_worker, args=(user_text, screen_context, False), daemon=True)
        active_llm_thread.start()

def auto_log_session_to_transcript():
    """Automatically exports the active session directly into the JSONL training dataset."""
    if not brain_node or not getattr(brain_node, 'history', None):
        return
        
    messages = []
    for msg in brain_node.history:
        role = msg.get("role", "")
        if role in ["user", "assistant", "system"]:
            messages.append({"role": role, "content": msg.get("content", "")})
            
    if messages:
        log_path = "training_logs.jsonl"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"messages": messages}) + "\n")
            print(f"[Transcript System] Automatically appended active turn block to {log_path}")
        except Exception as e:
            print(f"[Transcript System] Auto-log failed: {e}")

# ==========================================
# MAIN EXECUTION ENTRY POINT
# ==========================================
def main():
    global brain_node, voice_node, vts_node, eye_node, ear_node
    
    # --- FIXED: DEV TESTING OS INTERRUPT OVERRIDE ---
    # Captures Ctrl+C at the Windows kernel level, bypassing gevent entirely
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))
    # ------------------------------------------------
    
    print("\n=======================================================")
    print("Ada Local Intelligence Loop Initializing...")
    print("=======================================================")

    ear_node = Ear()
    brain_node = Brain()
    voice_node = Voice()
    vts_node = VTSBridge()
    eye_node = Eye()

    threading.Thread(target=background_audio_worker, args=(voice_node, audio_queue, telemetry_data, ada_state), daemon=True).start()
    threading.Thread(target=background_ear_worker, args=(ear_node, input_queue), daemon=True).start()
    threading.Thread(target=cognitive_loop, daemon=True).start()

    print("[System] Firing up the Graphical Interface...")
    base_web_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
    eel.init(base_web_path)
    
    # 1. Start Eel Server purely as a background network listener
    try:
        eel.start('index.html', mode=None, block=False, host='127.0.0.1', port=8686)
    except OSError:
        print("\n[FATAL] Port 8686 blocked! Run 'taskkill /F /IM python.exe' in terminal.")
        os._exit(1)
        
    # 2. Give the web server thread a clean second to claim the socket loop
    eel.sleep(1.0)
    
    # 3. FIXED WORKING METHOD: Direct path subprocess launch with isolated data profiles
    url = "http://127.0.0.1:8686/index.html"
    
    # Creates an isolated browser profile directory to strip window-size registry memory cache
    temp_profile_dir = os.path.abspath("./.ui_profile_cache")
    os.makedirs(temp_profile_dir, exist_ok=True)
    
    browser_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    
    app_opened = False
    for path in browser_paths:
        if os.path.exists(path):
            print(f"[System] Launching native UI container via: {path}")
            
            # --- FIXED: Added --user-data-dir to break the OS window size memory lock ---
            subprocess.Popen([
                path, 
                f"--app={url}", 
                "--window-size=1100,740", 
                f"--user-data-dir={temp_profile_dir}"
            ])
            app_opened = True
            break
            
    if not app_opened:
        print("[System] No local Chromium binary matched. Opening standard browser tab.")
        import webbrowser
        webbrowser.open(url)

    # 4. Keep script alive in background using an explicit, crash-proof loop
    print("\n[System] UI Dashboard Active. Core pipelines running.")
    print("[System] Closing the window will keep Ada running in the background.")
    print("[System] Use the UI 'Terminate System Pipeline' button to fully shut down.")
    
    while True:
        try:
            eel.sleep(1.0) # Maintains the active WebSocket listener background side
        except SystemExit:
            # Catching this prevents the window termination event from killing main.py
            print("\n[System] UI Window disconnected. Maintaining active background engines...")
            pass
        except KeyboardInterrupt:
            # --- FIXED: Dev Testing Bypass ---
            # Exits the local python process instantly but leaves Docker containers untouched!
            print("\n[System] Keyboard Interrupt detected. Exiting interface loop smoothly (Containers left running).")
            os._exit(0)

if __name__ == "__main__":
    try:
        ensure_local_models()
        ensure_vllm_containers()
        main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 60)
        print("FATAL PIPELINE CRASH DETECTED")
        traceback.print_exc()
        input("\nPress ENTER to close the window...")