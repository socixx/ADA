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
import urllib.request
from loguru import logger
from huggingface_hub import hf_hub_download, snapshot_download
import requests

# Scrub logger bloat
logger.remove()
logger.add(sys.stderr, level="WARNING") 

from features.brain import Brain
from features.voice import Voice
from features.ear import Ear
from features.vts_bridge import VTSBridge
from features.eye import Eye
import config

audio_queue = queue.Queue()
input_queue = queue.Queue()
telemetry_data = {"first_audio_timestamp": None}
ada_state = {"is_speaking": False, "last_generation": "", "spoken_text": ""}

def shutdown_containers():
    """Cleanly stops all background Docker nodes to free VRAM."""
    containers = [
        getattr(config, "LLM_CONTAINER_NAME", None),
        getattr(config, "VISION_CONTAINER_NAME", None),
        getattr(config, "VOXTRAL_CONTAINER_NAME", None),
    ]
    
    # Filter out None values
    target_containers = [c for c in containers if c]
    
    if target_containers and getattr(config, "LAUNCH_VLLM_CONTAINERS", False):
        print(f"\n[System] Instructing Docker to terminate nodes: {target_containers}...")
        subprocess.Popen(
            ["docker", "stop", "-t", "1"] + target_containers, 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        print("[System] VRAM release initiated.")

def ensure_local_models():
    """Pre-flight directory validation checks to ensure core assets exist."""
    active_tts = getattr(config, "ACTIVE_TTS", "KOKORO")
    
    # 1. LLM Verification
    llm_path = os.path.join(config.LOCAL_MODELS_DIR, config.LLM_MODEL)
    if os.path.exists(llm_path):
        print(f"[System] Verified local model matrix: {config.LLM_MODEL}")
    else:
        print(f"[System] ⚠️ LLM Matrix file missing at: {llm_path}")

    # 2. TTS Setup Branch Verification
    if active_tts == "KOKORO":
        kokoro_path = os.path.join(config.LOCAL_MODELS_DIR, "kokoro-v1.0.onnx")
        if os.path.exists(kokoro_path):
            print(f"[System] Verified local TTS matrix: kokoro-v1.0.onnx")
    elif active_tts == "QWEN_TTS":
        # FIX: Check for the new Qwen directory path layout maps
        qwen_dir = os.path.join(config.LOCAL_MODELS_DIR, "qwen3_tts_1.7b")
        if os.path.exists(os.path.join(qwen_dir, "config.json")):
            print(f"[System] Verified local TTS matrix: Qwen3-TTS Directory Status [Valid]")
        else:
            print(f"[System] Qwen3-TTS local weights missing. Initializing automatic synchronization loop...")

    print("\n[System] Initializing Ada Core (Native Monolithic Mode)")
    print("[System] Allocating VRAM for AI stack...")

    # --- 2. KOKORO ONNX CHECK ---
    kokoro_path = os.path.join(config.LOCAL_MODELS_DIR, "kokoro-v1.0.onnx")
    voices_path = os.path.join(config.LOCAL_MODELS_DIR, "voices-v1.0.bin")
    
    if not os.path.exists(kokoro_path) or not os.path.exists(voices_path):
        print("\n[System] Kokoro v1.0 ONNX assets not found. Downloading directly from GitHub releases...")
        try:
            import urllib.request
            
            print("[System] Downloading kokoro-v1.0.onnx...")
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", 
                kokoro_path
            )
            
            print("[System] Downloading voices-v1.0.bin...")
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", 
                voices_path
            )
            
            print("[System] ✅ Kokoro ONNX assets successfully acquired.\n")
        except Exception as e:
            print(f"[System] FATAL: Failed to download Kokoro ONNX payload: {e}")
            os._exit(1)
    else:
        print(f"[System] Verified local TTS matrix: kokoro-v1.0.onnx")

def get_container_status(container_name):
    """Queries the local Docker engine to check the current health status of a named node."""
    try:
        res = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Status}}", "-f", f"name={container_name}"],
            capture_output=True, text=True
        )
        return res.stdout.strip()
    except Exception:
        return ""

def wait_for_node(url, name, timeout=300):
    """Blocks execution loop until the target vLLM engine HTTP interface returns a healthy status."""
    start = time.time()
    print(f"[System] Waiting for {name} matrix validation...")
    while time.time() - start < timeout:
        try:
            res = requests.get(url, timeout=2)
            if res.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            time.sleep(2)
    print(f"\n[System] 🛑 FATAL: {name} container crashed or failed to start.")
    exit(1)

def ensure_local_models():
    """Pre-flight validation checks to confirm that core local model elements exist on disk."""
    active_tts = getattr(config, "ACTIVE_TTS", "KOKORO")
    
    # 1. LLM Verification
    llm_path = os.path.join(config.LOCAL_MODELS_DIR, config.LLM_MODEL)
    if os.path.exists(llm_path):
        print(f"[System] Verified local model matrix: {config.LLM_MODEL}")
    else:
        print(f"[System] ⚠️ LLM Matrix file missing at: {llm_path}")

    # 2. TTS Setup Branch Verification
    if active_tts == "KOKORO":
        kokoro_path = os.path.join(config.LOCAL_MODELS_DIR, "kokoro-v1.0.onnx")
        if os.path.exists(kokoro_path):
            print(f"[System] Verified local TTS matrix: kokoro-v1.0.onnx")
    elif active_tts == "QWEN_TTS":
        qwen_dir = os.path.join(config.LOCAL_MODELS_DIR, "qwen3_tts_1.7b")
        if os.path.exists(os.path.join(qwen_dir, "config.json")):
            print(f"[System] Verified local TTS matrix: Qwen3-TTS Directory Status [Valid]")
        else:
            print(f"[System] Qwen3-TTS local weights missing. Initializing automatic synchronization loop...")

    print("\n[System] Initializing Ada Core (Native Monolithic Mode)")
    print("[System] Allocating VRAM for AI stack...")

def ensure_vllm_containers():
    """Validates container orchestration layouts and hooks up runtime acceleration engines."""
    if not getattr(config, "LAUNCH_VLLM_CONTAINERS", True):
        return

    print("[System] Verifying local container runtime states...")

    # --- PHASE 1: COGNITIVE TEXT ENGINE NODE (Llama 3.1 GGUF vLLM Stack) ---
    status_llama = get_container_status(config.LLM_CONTAINER_NAME)
    if not status_llama:
        print(f"[System] Launching primary text architecture node: {config.LLM_CONTAINER_NAME}...")
        abs_checkpoints_dir = os.path.abspath(config.LOCAL_MODELS_DIR)
        
        cmd = [
            "docker", "run", "-d",
            "--name", config.LLM_CONTAINER_NAME,
            "--gpus", "all",
            "--ipc=host",
            "-p", f"{config.LLM_PORT}:8000",
            "-v", f"{config.HF_CACHE_DIR}:/root/.cache/huggingface",
            "-v", f"{abs_checkpoints_dir}:/app/checkpoints",
            "vllm/vllm-openai:latest",
            "--model", f"/app/checkpoints/{config.LLM_MODEL}",
            "--tokenizer", "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.60",
            "--max-model-len", "4096"
        ]
        subprocess.run(cmd)
    elif "Up" not in status_llama:
        print(f"[System] Waking primary text architecture node: {config.LLM_CONTAINER_NAME}")
        subprocess.run(["docker", "start", config.LLM_CONTAINER_NAME])

    wait_for_node(f"http://localhost:{config.LLM_PORT}/v1/models", "Text Engine (Llama)")

    # --- PHASE 3: AUDIO PROCESSING NODE (Custom Isolated ONNX Qwen3-TTS Container Stack) ---
    active_tts = getattr(config, "ACTIVE_TTS", "KOKORO")
    
    if active_tts == "KOKORO":
        target_model_dir = os.path.join(config.LOCAL_MODELS_DIR)
        kokoro_onnx_filename = "kokoro-v1.0.onnx"
        kokoro_onnx_path = os.path.join(target_model_dir, kokoro_onnx_filename)
        
        if not os.path.exists(kokoro_onnx_path):
            print("[System] Synchronizing stable Kokoro ONNX model weights...")
            os.makedirs(target_model_dir, exist_ok=True)
            hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=kokoro_onnx_filename, local_dir=target_model_dir)
            hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices.bin", local_dir=target_model_dir)
            print("✅ Kokoro ONNX assets fully verified on disk.")
            
    elif active_tts == "QWEN_TTS":
        model_folder_name = config.QWEN_MODEL.split("/")[-1].lower().replace("-", "_")
        target_model_dir = os.path.join(config.LOCAL_MODELS_DIR, model_folder_name)
        
        marker_config = os.path.join(target_model_dir, "config.json")
        if not os.path.exists(marker_config):
            print(f"[System] Snapshot marker missing. Synchronizing {config.QWEN_MODEL} assets...")
            os.makedirs(target_model_dir, exist_ok=True)
            snapshot_download(repo_id=config.QWEN_MODEL, local_dir=target_model_dir, ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
            print(f"✅ {config.QWEN_MODEL} matrix assets fully verified on disk.")

        status_qwen = get_container_status(config.QWEN_CONTAINER_NAME)
        if not status_qwen:
            print("[System] Compiling clean custom PyTorch TTS node container...")
            node_context_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "qwentts_node"))
            subprocess.run(["docker", "build", "-t", "ada-qwentts-node", node_context_dir])
            
            print(f"[System] Launching Isolated Custom Qwen-TTS Node Container: {config.QWEN_CONTAINER_NAME}...")
            abs_checkpoints_dir = os.path.abspath(target_model_dir)

            cmd = [
                "docker", "run", "-d", 
                "--name", config.QWEN_CONTAINER_NAME,
                "--gpus", "all",                      
                "--ipc=host",
                "-p", f"{config.QWEN_PORT}:8000",      
                "-v", f"{config.HF_CACHE_DIR}:/root/.cache/huggingface",
                "-v", f"{abs_checkpoints_dir}:/app/model",
                "ada-qwentts-node"
            ]
            subprocess.run(cmd)
        elif "Up" not in status_qwen:
            print(f"[System] Waking Optimized Audio node: {config.QWEN_CONTAINER_NAME}")
            subprocess.run(["docker", "start", config.QWEN_CONTAINER_NAME])

        # HEALTH VERIFICATION LOOP
        import requests
        print(f"[System] Waiting for Nano-vLLM Engine Pipeline initialization...")
        health_url = f"http://{config.QWEN_HOST}:{config.QWEN_PORT}/health"
        start_wait = time.time()
        engine_ready = False
        
        while time.time() - start_wait < 300: 
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200: 
                    print(f"\n✅ Production Nano-vLLM Audio Engine Active on port {config.QWEN_PORT}!")
                    engine_ready = True
                    break
            except requests.exceptions.RequestException:
                print(".", end="", flush=True)
                time.sleep(4)
                
        if not engine_ready:
            print("\n🛑 FATAL: Timeout boundary exceeded waiting for vLLM audio node backend.")
            exit(1)
        
    else:
        audio_container = getattr(config, "AUDIO_CONTAINER_NAME", "audio-engine-node")
        audio_port = getattr(config, "AUDIO_PORT", 8008)
        status_audio = get_container_status(audio_container)
        
        if not status_audio:
            print(f"[System] Building Local Audio Engine node (Kokoro): {audio_container}...")
            subprocess.run(["docker", "build", "-t", "ada-audio-node", "./audio_node"])
            cmd = [
                "docker", "run", "-d", "--name", audio_container, "--gpus", "all",
                "-v", f"{config.LOCAL_MODELS_DIR}:/models",
                "-p", f"{audio_port}:8008", 
                "ada-audio-node"
            ]
            subprocess.run(cmd)
        elif "Up" not in status_audio:
            print(f"[System] Waking Audio architecture node: {audio_container}")
            subprocess.run(["docker", "start", audio_container])

        wait_for_node("Audio Engine", audio_port, audio_container)

def background_audio_worker(voice, q, telemetry, state):
    is_first_sentence_of_turn = True
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break

        if item == "__RESET_TURN__":
            is_first_sentence_of_turn = True
            state["spoken_text"] = ""
            voice.reset_stop()
            q.task_done()
            continue

        sentence, queue_entry_time = item
        state["is_speaking"] = True
        
        _, _, internal_ttfa = voice.generate_voice_chunk(sentence)
        state["spoken_text"] += sentence + " "

        if is_first_sentence_of_turn:
            telemetry["first_audio_timestamp"] = queue_entry_time + (internal_ttfa if internal_ttfa else 0.0)
            is_first_sentence_of_turn = False

        state["is_speaking"] = False
        q.task_done()

def background_ear_worker(ear, in_q):
    while True:
        user_text, whisper_time, silence_start_timestamp = ear.listen_and_transcribe()
        if user_text.strip():
            in_q.put((user_text, whisper_time, silence_start_timestamp))

def main():
    print("\n[System] Initializing Ada Core (Native Monolithic Mode)")
    print("[System] Allocating VRAM for AI stack...")

    ear = Ear()
    brain = Brain()
    voice = Voice()
    vts = VTSBridge()
    eye = Eye()

    threading.Thread(
        target=background_audio_worker,
        args=(voice, audio_queue, telemetry_data, ada_state),
        daemon=True,
    ).start()

    threading.Thread(
        target=background_ear_worker,
        args=(ear, input_queue),
        daemon=True,
    ).start()

    print("\n=======================================================")
    print("Ada Local Intelligence Loop Fully Operational.")
    print("=======================================================")

    active_llm_thread = None

    try:
        while True:
            user_text, whisper_time, silence_start_timestamp = input_queue.get()
            print(f"\n[You]: {user_text}")

            if not audio_queue.empty() or ada_state["is_speaking"] or (active_llm_thread and active_llm_thread.is_alive()):
                print("[System] Interruption detected. Evaluating intent...")
                actual_spoken = ada_state["spoken_text"].strip()
                should_yield = brain.evaluate_interruption(actual_spoken, user_text)

                if should_yield:
                    print(f"[System] Ada yielded at: '{actual_spoken}'")
                    brain.abort_event.set()
                    voice.stop_with_fade(audio_queue)

                    if active_llm_thread and active_llm_thread.is_alive():
                        active_llm_thread.join(timeout=1.0)

                    brain.history.append({
                        "role": "assistant",
                        "content": actual_spoken + f"— [INTERRUPTED]",
                    })

                    if any(word in user_text.lower() for word in ["stop", "wait", "nevermind", "never mind", "hold on", "cancel"]):
                        user_text = f"*[Interrupts Ada]* {user_text} (System Directive: Acknowledge the interruption and DO NOT continue your previous thought. Drop the topic entirely.)"
                    else:
                        user_text = f"*[Interrupts Ada]* {user_text}"

                    print("[System] Brief processing pause...")
                    time.sleep(0.6)
                else:
                    print("[System] Ada persisted (backchannel ignored).")
                    continue

            clean_text = (
                user_text.lower()
                .replace(".", "").replace(",", "")
                .replace("!", "").replace("?", "")
                .strip()
            )
            
            if "good night ada" in clean_text:
                print("\n[System] Shutdown command recognized.")
                brain.abort_event.set()
                if active_llm_thread and active_llm_thread.is_alive():
                    active_llm_thread.join(timeout=1.0)
                
                voice.stop_with_fade(audio_queue)
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                    
                brain.abort_event.clear() 
                goodbye_msg = "Goodnight! Consolidating my memories now..."
                print(f"[Ada]: {goodbye_msg}")

                audio_queue.put("__RESET_TURN__")
                audio_queue.put((goodbye_msg, time.perf_counter()))
                audio_queue.join()

                brain.memory.deep_sleep_consolidation()
                shutdown_containers()
                print("[System] Offline. Goodnight.")
                os._exit(0) 

            trigger_phrases = [
                "look at my screen", "read my screen", "what am i looking at", 
                "what's on my screen", "can you see my screen", "what do you see",
                "check my screen", "look at this"
            ]
            screen_context = ""
            
            if any(phrase in clean_text for phrase in trigger_phrases):
                print("\n[System] Vision trigger recognized. Capturing framebuffer...")
                
                fillers = [
                    "*leans in* Let's see here...",
                    "*looks closely* Give me one sec...",
                    "*glances over* Let me check...",
                    "*squints* Hmm, looking now..."
                ]
                looking_msg = random.choice(fillers)
                
                audio_queue.put("__RESET_TURN__")
                audio_queue.put((looking_msg, time.perf_counter()))
                vts.trigger_action("looks at screen") 
                audio_queue.join() 
                
                vision_result = eye.look_at_screen()
                screen_context = f"\n[SYSTEM OBSERVATION: Ada just looked at the user's screen. She saw: {vision_result}]\n"
                print("[System] Vision payload synchronized into prompt context.")

            def llm_worker(text_to_process, visual_context, is_retry=False):
                telemetry_data["first_audio_timestamp"] = None
                brain.abort_event.clear()
                
                if not is_retry:
                    audio_queue.put("__RESET_TURN__")

                print(f"[Ada]: ", end="", flush=True)
                buffer = ""
                ada_state["last_generation"] = ""
                requires_auto_look = False
                
                # Configurable sentence window pool
                sentence_pool = []
                is_first_sentence = True
                target_window = getattr(config, "QWEN_SENTENCE_WINDOW", 2)

                for chunk in brain.get_response_stream(text_to_process, screen_context=visual_context):
                    if brain.abort_event.is_set(): break
                        
                    print(chunk, end="", flush=True)
                    buffer += chunk
                    ada_state["last_generation"] += chunk
                    
                    if any(p in chunk for p in ['.', '!', '?']):
                        idx = max(buffer.rfind('.'), buffer.rfind('!'), buffer.rfind('?'))
                        raw_sentence = buffer[: idx + 1]
                        buffer = buffer[idx + 1:]
                        
                        clean_sentence = raw_sentence.strip()
                        if clean_sentence:
                            actions = re.findall(r'\*(.*?)\*', clean_sentence)
                            for action in actions: 
                                vts.trigger_action(action.strip())
                            
                            speech_only = re.sub(r'\*.*?\*', '', clean_sentence).strip()
                            speech_only = speech_only.replace("'", "").replace('"', '').strip()
                            
                            if len(speech_only) > 1:
                                # Always dispatch the very first sentence immediately for instant TTFA
                                if is_first_sentence:
                                    audio_queue.put((speech_only, time.perf_counter()))
                                    is_first_sentence = False
                                else:
                                    sentence_pool.append(speech_only)
                                    if len(sentence_pool) >= target_window:
                                        combined_chunk = " ".join(sentence_pool)
                                        audio_queue.put((combined_chunk, time.perf_counter()))
                                        sentence_pool.clear()

                # Clean up trailing stream text fragments
                remaining_text = buffer.strip()
                if remaining_text and not brain.abort_event.is_set():
                    actions = re.findall(r'\*(.*?)\*', remaining_text)
                    for action in actions:
                        vts.trigger_action(action.strip())
                    
                    speech_only = re.sub(r'\*.*?\*', '', remaining_text).strip()
                    speech_only = speech_only.replace("[LOOK]", "").strip()
                    if speech_only:
                        sentence_pool.append(speech_only)

                if sentence_pool and not brain.abort_event.is_set():
                    combined_chunk = " ".join(sentence_pool)
                    audio_queue.put((combined_chunk, time.perf_counter()))
                
                print()
                
                if requires_auto_look:
                    print("\n[System] LLM requested visual confirmation. Auto-triggering camera...")
                    audio_queue.join()
                    
                    targeted_vision_result = eye.look_at_screen(f"Find and extract the exact details needed to answer this: {text_to_process}")
                    new_context = f"\n[SYSTEM OBSERVATION: Ada looked closely and extracted this data: {targeted_vision_result}]\n"
                    brain.history.append({"role": "system", "content": new_context})
                    
                    print("[System] Visual data retrieved. Re-prompting LLM with circuit breaker locked...")
                    follow_up_prompt = "(System: Answer the user's previous question strictly using the new SYSTEM OBSERVATION data. Be brief.)"
                    
                    threading.Thread(target=llm_worker, args=(follow_up_prompt, "", True), daemon=True).start()

            active_llm_thread = threading.Thread(target=llm_worker, args=(user_text, screen_context, False), daemon=True)
            active_llm_thread.start()

    except KeyboardInterrupt:
        print("\n[System] Shutting down pipeline. VRAM will be cleared.")
        shutdown_containers()
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
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        input("\nPress ENTER to close the window...")