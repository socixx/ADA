import mss
import io
import base64
from PIL import Image
from openai import OpenAI
import config

class Eye:
    def __init__(self):
        self.enabled = getattr(config, "ENABLE_VISION", True)
        if not self.enabled:
            print("[Eye] Vision disabled via config. Screen awareness offline.")
            self.client = None
            self.sct = None
            return
        print("[Eye] Connecting to local vLLM Vision Node (Port 8005)...")
        self.client = OpenAI(base_url=f"http://localhost:{config.VISION_PORT}/v1", api_key="token")
        self.sct = mss.mss()

    def look_at_screen(self, user_question="Scan the entire screen layout. Extract and list all visible text blocks, window titles, open applications, terminal outputs, active video metadata, and chat logs in full detail."):
        """Captures framebuffers and offloads dense visual analysis to the vLLM container."""
        if not self.enabled:
            return "[Vision offline — ENABLE_VISION is False in config.]"
        try:
            print("[Eye] Snapping screenshot...")
            monitor = self.sct.monitors[1]
            sct_img = self.sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # FULL RENDER DENSITY: Maintained layout sharpness for crisp text comprehension
            img.thumbnail((1280, 1280))
            
            # Encode raw binary frame details into standard base64 strings
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{img_base64}"
            
            # Post directly to the dedicated vLLM visual attention matrix
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2-VL-2B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_question},
                            {
                                "type": "image_url", 
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            output_text = response.choices[0].message.content
            print(f"[Eye] Vision Output: '{output_text.strip()}'")
            return output_text.strip()
            
        except Exception as e:
            print(f"[Eye] Capture failed: {e}")
            return "I tried to look, but my screen capture failed."