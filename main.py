import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import transformers
transformers.logging.set_verbosity_error()

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING") 

import time
import queue
import threading
import re
from features.brain import Brain
from features.voice import Voice
from features.ear import Ear
from features.vts_bridge import VTSBridge
from features.eye import Eye
import config
import random

audio_queue = queue.Queue()
input_queue = queue.Queue()
telemetry_data = {"first_audio_timestamp": None}

ada_state = {"is_speaking": False, "last_generation": "", "spoken_text": ""}

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
                        "content": actual_spoken + "— [INTERRUPTED]",
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
                print("[System] Offline. Goodnight.")
                os._exit(0) 

            # --- ON-DEMAND VISION INTERCEPTOR ---
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
                
                # DYNAMIC SYNC
                audio_queue.join() 
                vision_result = eye.look_at_screen()
                
                screen_context = f"\n[SYSTEM OBSERVATION: Ada just looked at the user's screen. She saw: {vision_result}]\n"
                print("[System] Vision payload synchronized into prompt context.")

            # --- NORMAL LLM GENERATION ---
            def llm_worker(text_to_process, visual_context, is_retry=False):
                telemetry_data["first_audio_timestamp"] = None
                brain.abort_event.clear()
                
                # Only clear the queue if it's a completely fresh user turn
                if not is_retry:
                    audio_queue.put("__RESET_TURN__")

                print(f"[Ada]: ", end="", flush=True)

                buffer = ""
                ada_state["last_generation"] = ""
                requires_auto_look = False

                for chunk in brain.get_response_stream(text_to_process, screen_context=visual_context):
                    if brain.abort_event.is_set():
                        break
                        
                    print(chunk, end="", flush=True)
                    buffer += chunk
                    ada_state["last_generation"] += chunk
                    
                    if "[LOOK]" in buffer:
                        # CIRCUIT BREAKER: If we are already on a retry, forbid another camera call
                        if is_retry:
                            buffer = buffer.replace("[LOOK]", "")
                        else:
                            requires_auto_look = True
                            brain.abort_event.set() 
                            break

                    while True:
                        boundaries = []
                        inside_square = False
                        inside_angle = False
                        inside_paren = False
                        inside_asterisk = False

                        for i, char in enumerate(buffer):
                            if char == '[': inside_square = True
                            elif char == ']': inside_square = False
                            elif char == '<': inside_angle = True
                            elif char == '>': inside_angle = False
                            elif char == '(': inside_paren = True
                            elif char == ')': inside_paren = False
                            elif char == '*': inside_asterisk = not inside_asterisk
                            
                            elif char in ['.', '!', '?', '\n', ',', ';', ':']:
                                if not (inside_square or inside_angle or inside_paren or inside_asterisk):
                                    boundaries.append(i)

                        if not boundaries:
                            break

                        idx = boundaries[0]
                        sentence = buffer[: idx + 1]
                        buffer = buffer[idx + 1:]

                        clean_sentence = sentence.strip()
                        if clean_sentence:
                            actions = re.findall(r'\*(.*?)\*', clean_sentence)
                            for action in actions:
                                if action.strip():
                                    vts.trigger_action(action.strip())
                                    
                            speech_only = re.sub(r'\*.*?\*', '', clean_sentence)
                            speech_only = speech_only.replace("[LOOK]", "").replace("...", ",").replace("..", ",").strip()
                            
                            if speech_only:
                                audio_queue.put((speech_only, time.perf_counter()))

                if buffer.strip() and not brain.abort_event.is_set():
                    actions = re.findall(r'\*(.*?)\*', buffer.strip())
                    for action in actions:
                        if action.strip():
                            vts.trigger_action(action.strip())
                            
                    speech_only = re.sub(r'\*.*?\*', '', buffer.strip())
                    speech_only = speech_only.replace("[LOOK]", "").replace("...", ",").replace("..", ",").strip()
                    if speech_only:
                        audio_queue.put((speech_only, time.perf_counter()))
                
                print()
                
                # --- THE AUTO-TRIGGER LOOP ---
                if requires_auto_look:
                    print("\n[System] LLM requested visual confirmation. Auto-triggering camera...")
                    
                    # Smart sync: Wait for the filler audio to clear out naturally
                    audio_queue.join()
                    
                    # Force the model to extract data specific to your question
                    targeted_vision_result = eye.look_at_screen(f"Find and extract the exact details needed to answer this: {text_to_process}")
                    
                    new_context = f"\n[SYSTEM OBSERVATION: Ada looked closely and extracted this data: {targeted_vision_result}]\n"
                    brain.history.append({"role": "system", "content": new_context})
                    
                    print("[System] Visual data retrieved. Re-prompting LLM with circuit breaker locked...")
                    
                    follow_up_prompt = "(System: Answer the user's previous question strictly using the new SYSTEM OBSERVATION data. Be brief.)"
                    
                    # Flag set to True to kill recursion loops completely
                    threading.Thread(target=llm_worker, args=(follow_up_prompt, "", True), daemon=True).start()

            active_llm_thread = threading.Thread(target=llm_worker, args=(user_text, screen_context, False), daemon=True)
            active_llm_thread.start()

    except KeyboardInterrupt:
        print("\n[System] Shutting down pipeline. VRAM will be cleared.")
        os._exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 60)
        print("FATAL PIPELINE CRASH DETECTED")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        input("\nPress ENTER to close the window...")