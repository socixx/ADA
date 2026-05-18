import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import config
from features.memory import MemoryManager

class Brain:
    def __init__(self):
        print(f"[Brain] Loading Native Llama 3.1 Model '{config.LLM_MODEL}' into VRAM...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.WHISPER_MODEL if "meta-llama" not in config.LLM_MODEL else config.LLM_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa" 
        )
        
        self.model_lock = threading.Lock()
        self.memory = MemoryManager(self.tokenizer, self.model, self.model_lock)
                
        self.history = []
        self.max_history_turns = 15 

    def evaluate_interruption(self, ada_current_text, user_interruption):
        prompt = (
            "You are a conversational dynamics engine. Ada is currently speaking. "
            "The user just interrupted her.\n\n"
            f"ADA WAS SAYING: \"{ada_current_text}\"\n"
            f"USER INTERRUPTED WITH: \"{user_interruption}\"\n\n"
            "RULES:\n"
            "1. If the user is just agreeing (e.g., 'yeah', 'right', 'mhmm'), output PERSIST.\n"
            "2. If the user asks a direct question, makes a correction, or says 'stop/wait', output YIELD.\n"
            "3. Output exactly ONE word: YIELD or PERSIST.\n\n"
            "DECISION:"
        )
        
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with self.model_lock:
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=5, temperature=0.1, pad_token_id=self.tokenizer.eos_token_id)
                
        decision = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
        return "YIELD" in decision

    def get_response_stream(self, text, chat_inbox=""):
        memory_payload = self.memory.load_memories(current_user_text=text)
        current_user_name = self.memory.user_name
        
        # THE FIX: We relax the strict 15-word limit for specific contexts
        system_prompt = (
            "You are Ada, a real human companion and VTuber. Speak casually, dryly, and concisely over voice chat.\n\n"
            f"{memory_payload}"
            f"[LIVE TWITCH CHAT INBOX]:\n{chat_inbox if chat_inbox else '(Chat is currently empty/offline)'}\n\n"
            "CRITICAL BEHAVIORAL CONSTRAINTS:\n"
            "1. NEVER use corporate AI phrases or break character.\n"
            "2. Keep standard conversational responses under 20 words. HOWEVER, if the user explicitly asks for a story, a detailed explanation, or a long narrative, you may ignore this limit and speak freely.\n"
            "3. DO NOT end every turn with a question.\n"
            f"4. Respond naturally to the User ({current_user_name}).\n"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.history:
            messages.append(turn)
            
        messages.append({"role": "user", "content": text})
        
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=300, temperature=0.6, top_p=0.9, pad_token_id=self.tokenizer.eos_token_id)
        
        def locked_generation():
            with self.model_lock:
                self.model.generate(**generation_kwargs)

        threading.Thread(target=locked_generation).start()
        
        assistant_response = ""
        for new_text in streamer:
            assistant_response += new_text
            yield new_text
            
        self.history.append({"role": "user", "content": text}) 
        self.history.append({"role": "assistant", "content": assistant_response.strip()})
        
        if len(self.history) > (self.max_history_turns * 2):
            discarded_chunk = self.history[:4]
            self.history = self.history[4:]
            self.memory.consolidate_to_timeline(discarded_chunk)
            
        self.memory.shadow_scribe_worker(text, self.history[:-1])