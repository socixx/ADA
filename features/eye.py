import mss
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Eye:
    def __init__(self):
        print("[Eye] Loading Qwen2-VL-2B Vision Scribe into VRAM...")
        model_id = "Qwen/Qwen2-VL-2B-Instruct" # Reverted to 2B for Fish Audio VRAM buffer
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda",
            attn_implementation="sdpa" 
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.sct = mss.mss()

    def look_at_screen(self, user_question="Act as an OCR script. Scan the entire screen layout. Extract and list all visible text blocks, open window titles, terminal commands, active video metadata, channel names, subscriber metrics, and chat logs. Provide a raw data dump."):
        try:
            print("[Eye] Snapping screenshot...")
            monitor = self.sct.monitors[1]
            sct_img = self.sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Keep it crisp at 1280x1280 so 2B can read tiny numbers/text labels cleanly
            img.thumbnail((1280, 1280))
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": img,
                            # Stripped max_pixels. The model now receives the raw high-fidelity grid.
                        },
                        {"type": "text", "text": user_question},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=200)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            print(f"[Eye] Vision Output: '{output_text.strip()}'")
            return output_text.strip()
            
        except Exception as e:
            print(f"[Eye] Capture failed: {e}")
            return "I tried to look, but my screen capture failed."