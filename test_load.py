print("🔍 Starting Monolithic Import Triage...")
import sys

print("1. Importing basic system utilities...")
import os
import time
import queue
import threading
import re
import warnings

print("2. Importing configuration framework...")
import config

print("3. Testing [Eye] (Qwen2-VL / Vision stack)...")
try:
    from features.eye import Eye
    print("   ✅ Eye imports cleanly.")
except Exception as e:
    print(f"   🛑 Eye threw an error: {e}")

print("4. Testing [VTS Bridge] (Live2D layout)...")
try:
    from features.vts_bridge import VTSBridge
    print("   ✅ VTS Bridge imports cleanly.")
except Exception as e:
    print(f"   🛑 VTS Bridge threw an error: {e}")

print("5. Testing [Ear] (Whisper / VAD layout)...")
try:
    from features.ear import Ear
    print("   ✅ Ear imports cleanly.")
except Exception as e:
    print(f"   🛑 Ear threw an error: {e}")

print("6. Testing [Voice] (Kokoro / Audio streams)...")
try:
    from features.voice import Voice
    print("   ✅ Voice imports cleanly.")
except Exception as e:
    print(f"   🛑 Voice threw an error: {e}")

print("7. Testing [Brain] (Llama 3.1 Quant engine)...")
try:
    from features.brain import Brain
    print("   ✅ Brain imports cleanly.")
except Exception as e:
    print(f"   🛑 Brain threw an error: {e}")

print("\n🎉 Verification pass complete!")