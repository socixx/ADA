import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import time
import os
import queue
import threading
from features.brain import Brain
from features.voice import Voice
from features.ear import Ear
import config

# Dual Queue System
audio_queue = queue.Queue()
input_queue = queue.Queue()
telemetry_data = {"first_audio_timestamp": None}

# Shared State
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
            # 1. Main thread waits for the Ear to catch something
            user_text, whisper_time, silence_start_timestamp = input_queue.get()
            print(f"\n[You]: {user_text}")

            # 2. BARGE-IN CHECK (Now checks if LLM is still generating too)
            if not audio_queue.empty() or ada_state["is_speaking"] or (active_llm_thread and active_llm_thread.is_alive()):
                print("[System] Interruption detected. Evaluating intent...")

                actual_spoken = ada_state["spoken_text"].strip()
                should_yield = brain.evaluate_interruption(actual_spoken, user_text)

                if should_yield:
                    print(f"[System] Ada yielded at: '{actual_spoken}'")

                    brain.abort_event.set()
                    voice.stop_with_fade(audio_queue)

                    # Wait a fraction of a second for the old thread to safely die
                    if active_llm_thread and active_llm_thread.is_alive():
                        active_llm_thread.join()

                    # Append the interrupted state to history
                    brain.history.append({
                        "role": "assistant",
                        "content": actual_spoken + "— [INTERRUPTED]",
                    })

                    # Context Bleed Fix
                    if any(word in user_text.lower() for word in ["stop", "wait", "nevermind", "never mind", "hold on", "cancel"]):
                        user_text = f"*[Interrupts Ada]* {user_text} (System Directive: Acknowledge the interruption and DO NOT continue your previous thought. Drop the topic entirely.)"
                    else:
                        user_text = f"*[Interrupts Ada]* {user_text}"

                    print("[System] Brief processing pause...")
                    time.sleep(0.6)

                else:
                    print("[System] Ada persisted (backchannel ignored).")
                    continue

            # 3. SHUTDOWN INTERCEPTOR
            clean_text = (
                user_text.lower()
                .replace(".", "").replace(",", "")
                .replace("!", "").replace("?", "")
                .strip()
            )
            if "good night ada" in clean_text:
                print("\n[System] 🛑 Shutdown command recognized.")
                
                # FIX: Kill any active LLM threads to release the model_lock!
                brain.abort_event.set()
                if active_llm_thread and active_llm_thread.is_alive():
                    active_llm_thread.join(timeout=1.0)
                
                # Clear the queue of any lingering audio
                voice.stop_with_fade(audio_queue)
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                    
                brain.abort_event.clear() # Reset so she can say goodnight
                
                goodbye_msg = "Goodnight! Consolidating my memories now..."
                print(f"[Ada]: {goodbye_msg}")

                audio_queue.put("__RESET_TURN__")
                audio_queue.put((goodbye_msg, time.perf_counter()))
                audio_queue.join()

                # Lock is now guaranteed to be free
                brain.memory.deep_sleep_consolidation()
                print("[System] Offline. Goodnight.")
                os._exit(0)


            # 4. NORMAL LLM GENERATION (Now spawned as a daemon thread)
            def llm_worker(text_to_process):
                telemetry_data["first_audio_timestamp"] = None
                brain.abort_event.clear()
                audio_queue.put("__RESET_TURN__")

                print(f"[Ada]: ", end="", flush=True)

                buffer = ""
                ada_state["last_generation"] = ""

                for chunk in brain.get_response_stream(text_to_process):
                    print(chunk, end="", flush=True)
                    buffer += chunk
                    ada_state["last_generation"] += chunk

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
                            audio_queue.put((clean_sentence, time.perf_counter()))

                if buffer.strip() and not brain.abort_event.is_set():
                    audio_queue.put((buffer.strip(), time.perf_counter()))
                
                print()

            active_llm_thread = threading.Thread(target=llm_worker, args=(user_text,), daemon=True)
            active_llm_thread.start()

    except KeyboardInterrupt:
        print("\n[System] Shutting down pipeline. VRAM will be cleared.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 60)
        print("🛑 FATAL PIPELINE CRASH DETECTED 🛑")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        input("\nPress ENTER to close the window...")