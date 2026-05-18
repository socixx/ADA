import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import time
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

# Updated Shared State
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
            state["spoken_text"] = "" # Reset what she's spoken for the new turn
            q.task_done()
            continue
            
        sentence, queue_entry_time = item
        
        state["is_speaking"] = True
        _, _, internal_ttfa = voice.generate_voice_chunk(sentence)
        
        # MAGIC FIX 1: Only log the text as "remembered" after it physically plays out the speakers
        state["spoken_text"] += sentence + " "
        
        if is_first_sentence_of_turn:
            telemetry["first_audio_timestamp"] = queue_entry_time + (internal_ttfa if internal_ttfa else 0.0)
            is_first_sentence_of_turn = False
            
        state["is_speaking"] = False
        q.task_done()

def background_ear_worker(ear, in_q):
    """Continuously listens and pushes transcribed text, even while Ada is talking."""
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

    # Start the Voice Thread
    threading.Thread(target=background_audio_worker, args=(voice, audio_queue, telemetry_data, ada_state), daemon=True).start()
    
    # Start the Ear Thread (Always Listening)
    threading.Thread(target=background_ear_worker, args=(ear, input_queue), daemon=True).start()

    print("\n=======================================================")
    print("Ada Local Intelligence Loop Fully Operational.")
    print("=======================================================")
    
    try:
        while True:
            # 1. Main thread waits for the Ear to catch something
            user_text, whisper_time, silence_start_timestamp = input_queue.get()
            print(f"\n[You]: {user_text}")
            
            # 2. IS ADA CURRENTLY SPEAKING? (The Barge-In Check)
            if not audio_queue.empty() or ada_state["is_speaking"]:
                print("[System] Interruption Detected. Evaluating semantic intent...")
                
                # We evaluate against what she actually said out loud, not what she generated
                actual_spoken = ada_state["spoken_text"].strip()
                should_yield = brain.evaluate_interruption(actual_spoken, user_text)
                
                if should_yield:
                    print(f"[System] Ada Yielded at: '{actual_spoken}'")
                    
                    # Trigger the acoustic trail-off. 
                    # This gracefully fades the audio and flushes the queue simultaneously.
                    if hasattr(voice, 'stop_with_fade'):
                        voice.stop_with_fade(audio_queue)
                    else:
                        with audio_queue.mutex:
                            audio_queue.queue.clear()
                        
                    # C. Slice context using ONLY what made it out of the speakers
                    brain.history.pop() # Remove the full generation that she never got to finish
                    brain.history.append({"role": "assistant", "content": actual_spoken + "— [INTERRUPTED]"})
                    
                    # MAGIC FIX 2: Tag your text so the LLM knows you barged in
                    user_text = f"*[Interrupts Ada]* {user_text}"
                    
                    # MAGIC FIX 3: The "Processing" Pause. 
                    # This prevents the jarring instant-snap and simulates her listening to the rest of your interruption.
                    print("[System] Pausing slightly to simulate human conversational processing...")
                    time.sleep(0.8) 
                    
                else:
                    print("[System] Ada Persisted (Backchannel ignored).")
                    continue
            
            # 3. SHUTDOWN INTERCEPTOR
            clean_text = user_text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()
            if "good night ada" in clean_text:
                print("\n[System] 🛑 Shutdown command recognized.")
                goodbye_msg = "Goodnight! Consolidating my memories now..."
                print(f"[Ada]: {goodbye_msg}")
                
                audio_queue.put("__RESET_TURN__")
                audio_queue.put((goodbye_msg, time.perf_counter()))
                audio_queue.join() 
                
                brain.memory.deep_sleep_consolidation()
                print("[System] Offline. Goodnight.")
                sys.exit(0)
            
            # 4. NORMAL LLM GENERATION
            telemetry_data["first_audio_timestamp"] = None
            audio_queue.put("__RESET_TURN__")
            
            print(f"[Ada]: ", end="", flush=True)
            
            llm_first_chunk_time = 0.0
            llm_first_clause_time = 0.0
            first_chunk = True
            first_clause = True
            llm_start_t = time.perf_counter()
            
            buffer = ""
            ada_state["last_generation"] = "" # Reset the tracking buffer for the new turn
            
            for chunk in brain.get_response_stream(user_text):
                if first_chunk:
                    llm_first_chunk_time = time.perf_counter() - llm_start_t
                    first_chunk = False
                
                print(chunk, end="", flush=True)
                buffer += chunk
                ada_state["last_generation"] += chunk # Track what she has generated so far
                
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
                    sentence = buffer[:idx + 1]
                    buffer = buffer[idx + 1:]
                    
                    clean_sentence = sentence.strip()
                    if clean_sentence:
                        if first_clause:
                            llm_first_clause_time = time.perf_counter() - llm_start_t
                            first_clause = False
                        audio_queue.put((clean_sentence, time.perf_counter()))
                        
            if buffer.strip():
                if first_clause:
                    llm_first_clause_time = time.perf_counter() - llm_start_t
                    first_clause = False
                audio_queue.put((buffer.strip(), time.perf_counter()))
                
            print() 
            # We NO LONGER join the audio queue here! 
            # If we block the main thread waiting for audio to finish, the Ear can't process interruptions.
            # We let the Ear queue handle the blocking at the top of the while loop.
                
    except KeyboardInterrupt:
        print("\n[System] Shutting down pipeline. VRAM will be cleared.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("🛑 FATAL PIPELINE CRASH DETECTED 🛑")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        input("\nPress ENTER to close the window...")