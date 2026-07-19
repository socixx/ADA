The Architecture: Sensory Fusion Engine (vision_scribe.py)
This engine runs entirely as a detached background worker. Its sole purpose is to constantly update a shared dictionary state (Live_Workspace_State) at 5 ticks per second. When Ada's main cognitive loop fires, she instantly reads this variable with zero added latency.

Pillar 1: Native OS Telemetry (The Global Context)
Tech Stack: win32gui, psutil.

Execution: Polls every 0.5s. Costs 0 VRAM.

Role: Instantly tells Ada the active application. If you switch from VS Code to Minecraft 1.21.4, she immediately shifts her conversational context before you even speak.

Pillar 2: UIAutomation (The Coding Fast-Path)
Tech Stack: uiautomation.

Execution: Event-driven. Costs 0 VRAM.

Role: Hooked directly into Windows, this grabs whatever raw text your mouse is currently hovering over or highlighting. If you highlight a broken Python function and say "Why is this throwing an error?", the exact string of code is already in her ticker.

Pillar 3: Attention Tracker & Micro-OCR (The Universal Fallback)
Tech Stack: pynput + winrt.windows.media.ocr.

Execution: ~15ms per trigger. Costs 0 VRAM.

Role: For applications that block UIAutomation (like game engines). pynput tracks your cursor's X/Y coordinates. If you draw a circle or hover over an area, it snaps a 400x400 pixel bounding box around the cursor and feeds it to Windows 11's native built-in OCR. It instantly reads the coordinate text, inventory item, or stat sheet you are physically pointing at.

Pillar 4: Desktop Audio Loopback (Media Comprehension)
Tech Stack: soundcard (WASAPI loopback) + Lightweight local STT (Faster-Whisper Tiny/Base).

Execution: Continuous rolling buffer.

Role: Listens exclusively to your desktop audio output, distinct from your mic input. If you are watching a video or an NPC is talking in a game, Ada receives a live transcript of their dialogue, allowing her to react to the media alongside you.

Pillar 5: Moondream2 (The Visual "Vibe Checker")
Tech Stack: transformers (vikhyatk/moondream2).

Execution: 1 frame every 2-3 seconds. Uses minimal VRAM (~3-4GB in fp16).

Role: The ultimate scene describer. Because it is highly optimized (1.8B parameters) and runs asynchronously, it constantly generates broad English captions of your screen (e.g., "A blocky landscape featuring a dark cave, a lava pool, and a character holding a diamond pickaxe."). This provides the visual intuition that OCR and text-scraping miss.

The Data Flow: The Context Ticker
All 5 pillars feed into a single, rolling text variable. At any given moment, the Live_Workspace_State looks exactly like this:

Plaintext
[LIVE WORKSPACE STATE]
Active App: "Minecraft 1.21"
Highlighted Text (UIA): [None]
Attention Area (OCR): "Diamond Pickaxe", "Durability: 45/1561"
Visual Scene (Moondream2): "A dark underground cave next to a pool of glowing lava."
Desktop Audio: *Lava popping*, *Zombie groans*
The Interaction Loop:

You circle your cursor over your tool and say: "Do I have enough durability to mine this obsidian?"

Ada's brain.py grabs your transcription and silently prepends the Live_Workspace_State to the LLM prompt.

Ada reads the text, sees the durability stats from the OCR, sees the lava from Moondream2, and instantly replies: "You only have 45 uses left on that pickaxe. Plus, there's lava right there—don't risk dropping it if it breaks."

Implementation Roadmap
To transition to this gracefully without breaking your current build, we will deprecate the old features/eye.py and roll this out in three distinct phases:

Phase 1: The Zero-VRAM Foundation
Create features/vision_scribe.py.

Implement Pillar 1 (win32gui) and Pillar 2 (UIAutomation).

Wire it into main.py as a background thread (threading.Thread(target=vision_scribe_worker...)).

Result: Ada instantly gains the ability to read your active coding windows and highlighted text with zero latency.

Phase 2: The Attention & Scene Generators
Add Pillar 3 (pynput + WinRT OCR) to track the mouse and grab targeted text in games.

Spin up a lightweight local inference node for Pillar 5 (Moondream2) to start pumping visual descriptions into the background loop.

Phase 3: The Audio Bridge
Add Pillar 4 (WASAPI Loopback) to capture and transcribe desktop audio.