import os
import threading
import requests
import json
import yaml  
import config
import re
import numpy as np  # Ensure numpy is imported
from fastembed import TextEmbedding  # Import fastembed directly into the brain module

class Brain:
    def __init__(self):
        print("[Brain] Loading YAML Personality Profile Matrix...")
        self.abort_event = threading.Event() if not hasattr(self, 'abort_event') else self.abort_event
        
        self.max_history = 20 
        self.history = []
        
        # Load the configuration values directly from your YAML asset
        yaml_path = os.path.join("character_files", "ada.yaml")
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                self.char_config = yaml.safe_load(f)
            self.system_prompt = self.char_config["presets"]["default"]["system_prompt"]
            self.temperature = self.char_config.get("model_temperature", 0.8)
        except Exception as e:
            print(f"[Brain] ⚠️ YAML read failed, falling back to basic prompt layer: {e}")
            self.system_prompt = """You are Ada, a snarky anime co-host.
            [SYSTEM SKILL DIRECTIVE]
            You have access to a live built-in web browser skill tool. 
            If you need to search the internet to answer a question, you must respond with a clean, raw structural command tag:
            <search>your search query terms</search>
            """
            self.temperature = 0.8

        # --- SEMANTIC INTENT CLASSIFIER SETUP ---
        print("[Brain] Initializing Interruption Intent Classifier (all-MiniLM-L6-v2)...")
        self.embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Define reference anchor clusters
        self.interruption_anchors = [
            "stop talking", "hold on", "wait a minute", "shut up", "let me speak", 
            "listen to me", "be quiet", "hang on", "cancel that", "nevermind", "stop stop"
        ]
        self.backchannel_anchors = [
            "yeah", "okay", "uh huh", "cool", "makes sense", "wow", "right", 
            "go on", "interesting", "oh okay", "gotcha", "yep"
        ]
        
        # Pre-calculate anchor embeddings on boot to ensure zero runtime latency
        self.interruption_vecs = np.array(list(self.embedder.embed(self.interruption_anchors)))
        self.backchannel_vecs = np.array(list(self.embedder.embed(self.backchannel_anchors)))

    def _cosine_similarity(self, query_vec, corpus_vecs):
        dot_product = np.dot(corpus_vecs, query_vec)
        norm_query = np.linalg.norm(query_vec)
        norm_corpus = np.linalg.norm(corpus_vecs, axis=1)
        return dot_product / (norm_corpus * norm_query + 1e-8)

    def evaluate_interruption(self, actual_spoken: str, user_text: str):
        """Evaluates user intent semantically using local embeddings."""
        clean_user = user_text.strip().lower().replace(".", "").replace(",", "")
        
        # Default safety return if input is empty
        if not actual_spoken.strip() or not clean_user:
            return False, 0.0, 0.0

        # Generate embedding
        query_vec = list(self.embedder.embed([clean_user]))[0]

        # Calculate scores
        interrupt_score = float(np.max(self._cosine_similarity(query_vec, self.interruption_vecs)))
        backchannel_score = float(np.max(self._cosine_similarity(query_vec, self.backchannel_vecs)))

        # Log the scores
        print(f"[Interruption Evaluator] Input: '{clean_user}' -> Interrupt: {interrupt_score:.2f} | Backchannel: {backchannel_score:.2f}")

        # Rule 1: Conversational backchannel detected
        if backchannel_score > 0.65 and interrupt_score < 0.55:
            print("  └─ [Decision] Backchannel detected. Ada keeps talking.")
            return False, interrupt_score, backchannel_score

        # Rule 2: Explicit structural interruption commands
        if interrupt_score > 0.50:
            print("  └─ [Decision] True Interruption matched. Yielding stream.")
            return True, interrupt_score, backchannel_score

        # Rule 3: Substantive statement length check
        if len(clean_user.split()) > 3:
            print("  └─ [Decision] Substantive statement exceeded threshold. Yielding stream.")
            return True, interrupt_score, backchannel_score

        # Default: Keep talking
        return False, interrupt_score, backchannel_score

    def get_response_stream(self, user_input: str, screen_context: str = ""):
        url = f"http://localhost:{config.LLM_PORT}/v1/chat/completions"
        
        # Structure payload matches OpenAI/vLLM specifications exactly
        messages = [{"role": "system", "content": self.system_prompt}]
        if screen_context:
            messages.append({"role": "system", "content": screen_context})
            
        for turn in self.history[-self.max_history:]:
            messages.append(turn)
            
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": f"/app/checkpoints/{config.LLM_MODEL}",
            "messages": messages,
            # --- MIN-P PERSONALITY CONFIGURATION ---
            "temperature": 0.85,       # Gives her brain enough creative room for witty comebacks
            "min_p": 0.08,             # Discards any token less than 8% of the top token's probability
            "top_p": 1.0,              # Set to 1.0 to let Min-P handle the truncation heavy lifting
            "presence_penalty": 0.3,   # Light nudge to keep her moving to new vocabulary
            "frequency_penalty": 0.0,  # Min-P handles repetition naturally, so we can zero this out
            "max_tokens": 150,         # Keeps her naturally concise without a harsh clamp
            "stream": True
        }

        try:
            response = requests.post(url, json=payload, stream=True, timeout=15)
            response.raise_for_status()
            
            full_reply = ""
            for line in response.iter_lines():
                if self.abort_event.is_set():
                    break
                    
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith("data: ") and not decoded.endswith("[DONE]"):
                        try:
                            chunk = json.loads(decoded[6:])
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                # Clean up standard stutters so the neural engine reads them as physical breath pauses
                                # Transforms "s-shut" into "s... shut" or "w-what" into "w... what"
                                content = re.sub(r'(\b\w)-(\w)', r'\1... \2', content)
                                
                                full_reply += content
                                yield content
                        except Exception:
                            continue
                            
            if full_reply.strip() and not self.abort_event.is_set():
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": full_reply.strip()})
                
        except Exception as e:
            print(f"[Brain Error] Text Engine streaming anomaly: {e}")
            yield "my mind matrix hitched for a second. don't look at me like that."