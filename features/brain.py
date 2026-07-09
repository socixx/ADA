import os
import threading
import requests
import json
import yaml  # Make sure to run: pip install pyyaml
import config
import re

class Brain:
    def __init__(self):
        print("[Brain] Loading YAML Personality Profile Matrix...")
        self.abort_event = threading.Event() if not hasattr(self, 'abort_event') else self.abort_event
        
        # Consistent message history tracking matching Riko's implementation pattern
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
            self.system_prompt = "You are Ada, a snarky anime co-host."
            self.temperature = 0.8

    def evaluate_interruption(self, actual_spoken: str, user_text: str) -> bool:
        if not actual_spoken.strip():
            return False
        stop_signals = ["stop", "wait", "hold on", "shut up", "cancel", "nevermind"]
        if any(word in user_text.lower() for word in stop_signals):
            return True
        return len(user_text.strip().split()) > 2

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