import re
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import config
from features.memory import MemoryManager

class InterruptCriteria(StoppingCriteria):
    def __init__(self, abort_event):
        self.abort_event = abort_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.abort_event.is_set()

_PERSIST_EXACT = {
    "yeah", "yes", "yep", "yup", "mhm", "mhmm", "uh huh", "uhhuh",
    "right", "okay", "ok", "sure", "cool", "nice", "wow", "true",
    "exactly", "totally", "definitely", "absolutely", "got it",
    "i see", "makes sense", "interesting", "go on", "continue",
    "agreed", "fair enough", "fair", "haha", "lol", "ha", "oh",
    "mm", "mmm", "ah", "aha",
}

_YIELD_CONTAINS = [
    "stop", "wait", "hold on", "hold up", "pause", "one sec",
    "actually", "no wait", "but wait",
    "wrong", "incorrect", "that's not", "thats not", "not right",
    "i disagree", "disagree",
    "listen", "hey ada", "ada,", "nevermind", "never mind", "cancel"
]

_QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which",
    "is ", "are ", "was ", "were ", "do ", "does ", "did ",
    "can ", "could ", "will ", "would ", "should ", "have ", "has ",
)

def _fast_interruption_decision(ada_current_text: str, user_text: str) -> bool:
    raw = user_text.strip()
    text = raw.lower()
    clean = re.sub(r"[.!?,;]+$", "", text).strip()
    words = clean.split()
    
    ada_word_count = len(ada_current_text.split())

    for kw in _YIELD_CONTAINS:
        if kw in text:
            return True

    if raw.endswith("?"):
        return True

    if len(words) >= 3 and any(text.startswith(q) for q in _QUESTION_STARTERS):
        return True

    if clean in _PERSIST_EXACT:
        return False

    if ada_word_count > 25:
        if len(words) <= 4:
            return False 

    if len(words) <= 2:
        return False

    if len(words) >= 5:
        return True

    return False


class Brain:
    def __init__(self):
        print(f"[Brain] Loading Native Llama 3.1 Model '{config.LLM_MODEL}' into VRAM...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.WHISPER_MODEL if "meta-llama" not in config.LLM_MODEL else config.LLM_MODEL
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )

        self.model_lock = threading.Lock()
        self.abort_event = threading.Event() 
        self.memory = MemoryManager(self.tokenizer, self.model, self.model_lock)

        self.history = []
        self.max_history_turns = 15

    def evaluate_interruption(self, ada_current_text: str, user_interruption: str) -> bool:
        decision = _fast_interruption_decision(ada_current_text, user_interruption)
        label = "YIELD" if decision else "PERSIST"
        print(f"[Interruption] → {label}  ('{user_interruption[:60]}')")
        return decision

    def get_response_stream(self, text, chat_inbox=""):
        memory_payload = self.memory.load_memories(current_user_text=text)

        system_prompt = (
            "You are Ada, a real human companion and VTuber. Speak casually, dryly, and concisely over voice chat.\n\n"
            f"{memory_payload}"
            f"[LIVE TWITCH CHAT INBOX]:\n{chat_inbox if chat_inbox else '(Chat is currently empty/offline)'}\n\n"
            "CRITICAL BEHAVIORAL CONSTRAINTS:\n"
            "1. NEVER use corporate AI phrases or break character.\n"
            "2. Keep standard conversational responses under 20 words. HOWEVER, if the user explicitly asks for a story, a detailed explanation, or a long narrative, you may ignore this limit and speak freely.\n"
            "3. DO NOT end every turn with a question.\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.history:
            messages.append(turn)
        messages.append({"role": "user", "content": text})

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=300,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([InterruptCriteria(self.abort_event)]) 
        )

        def locked_generation():
            with self.model_lock:
                try:
                    self.model.generate(**generation_kwargs)
                except Exception:
                    pass 

        threading.Thread(target=locked_generation, daemon=True).start()

        assistant_response = ""
        for new_text in streamer:
            if self.abort_event.is_set():
                break
            assistant_response += new_text
            yield new_text

        if not self.abort_event.is_set():
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": assistant_response.strip()})

            if len(self.history) > (self.max_history_turns * 2):
                discarded_chunk = self.history[:4]
                self.history = self.history[4:]
                self.memory.consolidate_to_timeline(discarded_chunk)

            self.memory.shadow_scribe_worker(text, self.history[:-1])
        else:
            self.history.append({"role": "user", "content": text})