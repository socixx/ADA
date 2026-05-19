import threading
from openai import OpenAI
import config
from features.memory import MemoryManager

class Brain:
    def __init__(self):
        print("[Brain] Connecting to local vLLM Text Node (Port 8000)...")
        # Lightweight client interface -> Uses 0MB of local Windows VRAM
        self.client = OpenAI(base_url=f"http://localhost:{config.LLM_PORT}/v1", api_key="token")
        
        self.history = []
        self.abort_event = threading.Event()
        self.model_lock = threading.Lock()
        
        # Binds into your existing long-term profile memory layers
        self.memory = MemoryManager(None, None, self.model_lock) 

    def evaluate_interruption(self, actual_spoken, user_text):
        """Evaluates whether Ada should yield the turn during an incoming voice interruption."""
        if not actual_spoken.strip():
            return True
            
        system_prompt = (
            "You are the sub-conscious timing coordinate system of an AI companion named Ada.\n"
            "Analyze the conversation state and decide if the user's new input requires you to STOP speaking immediately.\n"
            "Respond ONLY with 'YIELD' or 'PERSIST'. Do not write sentences.\n\n"
            "Rules:\n"
            "- YIELD if the user changes the subject, asks a direct question, or explicitly tells you to stop.\n"
            "- PERSIST if the user is just saying background noises, short agreements ('yeah', 'cool'), or accidental audio spikes."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Ada was saying: '{actual_spoken}'\nUser suddenly cut in with: '{user_text}'"}
        ]
        
        try:
            with self.model_lock:
                response = self.client.chat.completions.create(
                    model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=5
                )
            decision = response.choices[0].message.content.strip().upper()
            return "YIELD" in decision
        except Exception as e:
            print(f"[Interruption Evaluation Fail] Defaulting to yield: {e}")
            return True

    def get_response_stream(self, user_text, chat_inbox=None, screen_context=""):
        """Streams text chunks in real-time straight from the local vLLM engine."""
        memory_payload = self.memory.load_memories(user_text)
        
        system_prompt = (
            "You are Ada, a real human companion and VTuber. Speak casually, dryly, and naturally over voice chat.\n\n"
            f"{memory_payload}"
            f"{screen_context}\n"
            f"[LIVE TWITCH CHAT INBOX]:\n{chat_inbox if chat_inbox else '(Chat is currently empty/offline)'}\n\n"
            "CRITICAL BEHAVIORAL CONSTRAINTS:\n"
            "1. NEVER use corporate AI phrases or break character.\n"
            "2. Speak completely naturally without any artificial word count constraints or length limits. Express your thoughts with full depth whenever a topic or data list warrants it.\n"
            "3. DO NOT end every turn with a question.\n"
            "4. Express emotions and physical actions using asterisks (e.g., *smiles*, *laughs*, *rolls eyes*).\n"
            "5. SCREEN AWARENESS: When given [SYSTEM OBSERVATION] logs, synthesize the raw data naturally. Do not repeat the data back as a dry list. React to what the user is doing or watching like a casual friend hanging out. If data is missing and you cannot answer a specific question, output EXACTLY the tag [LOOK] with a short filler phrase so your camera triggers.\n"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.history[-6:]:  # Rolling conversational context window
            messages.append(turn)
        messages.append({"role": "user", "content": user_text})
        
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,  # <--- Now dynamically linked to config.py
                messages=messages,
                temperature=0.7,
                stream=True
            )
            
            for chunk in response:
                if self.abort_event.is_set():
                    break
                content = chunk.choices[0].delta.content
                if content:
                    yield content
                    
        except Exception as e:
            print(f"[Brain Stream Error] {e}")
            yield "My local backend connection hiccuped."