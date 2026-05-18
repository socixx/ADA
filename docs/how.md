# Ada Core Intelligence Architecture: Technical Specification

## Overview

The Ada Core is a fully local, native monolithic AI system designed to run on high-end consumer hardware (such as an RTX 3090). It mimics biological cognitive processes by fusing real-time acoustic processing, a localized Large Language Model (Llama 3.1 8B), a zero-latency audio synthesis pipeline, and a highly advanced four-tier memory architecture.

The system operates entirely offline, ensuring absolute privacy, zero network latency, and continuous autonomous operation.

---

## 1. The Acoustic Pipeline (Ear & Voice)

The system’s sensory input and output are governed by ultra-low-latency neural models rather than cloud APIs.

### The Input (The Ear)

To prevent the LLM from constantly analyzing background noise, the listening pipeline utilizes a dual-gate architecture:

* **The Wake Gate (Acoustic VAD):** Powered by **Silero VAD**, this is a lightweight neural network that monitors microphone input for human vocal frequencies. The system remains completely dormant until it physically hears speech.
* **The Cutoff Gate (Semantic VAD):** Powered by **Pipecat Smart-Turn**, this gate solves the "awkward pause" problem. Instead of relying purely on acoustic silence (which triggers prematurely if the user takes a breath), the Semantic VAD reads the live transcription and mathematically evaluates if the sentence is grammatically complete before ending the turn.
* **Transcription:** **Whisper STT** (Distil-Medium) converts the audio buffer to text instantly upon the Cutoff Gate triggering.

### The Output (The Voice)

* **Native TTS Engine:** Powered by **Kokoro-82M**, loaded natively into VRAM.
* **CUDA Graph Caching:** Upon startup, the engine undergoes a "warmup" sequence, pre-compiling its execution graph in VRAM to eliminate the "first-turn lag" typical of local TTS engines.
* **Numpy Audio Arrays:** Kokoro generates raw mathematical NumPy audio arrays at 24,000 Hz, which are streamed directly to the hardware speakers via `sounddevice`.

---

## 2. The Generation Pipeline (Producer/Consumer Queue)

To bridge the gap between text generation speed (which is fast) and audio synthesis speed (which is slower), the architecture utilizes a **Triple-Queue Asynchronous Pipeline** to achieve zero-latency conversational flow.

1. **Text Queue (LLM):** Llama 3.1 streams output tokens. A custom parser watches for sentence-ending punctuation (`.`, `!`, `?`). As soon as a full clause is detected, it is dropped into the queue.
2. **Synthesis Queue (Producer):** A background audio worker instantly grabs that sentence and feeds it to Kokoro, generating the NumPy audio array silently in the background.
3. **Playback Queue (Consumer):** A dedicated playback thread grabs the pre-generated audio arrays and plays them back-to-back.

This asynchronous design ensures that while the user is listening to Sentence 1, the GPU is already synthesizing Sentence 2, completely masking the compute time and eliminating robotic pauses.

---

## 3. The Four-Tier Biological Memory System

The crown jewel of the architecture is its state-aware, multi-tiered memory system. By categorizing data into specific physical spaces and lifespans, the system prevents context window bloat while maintaining deep, persistent knowledge.

### Tier 1: Short-Term Memory (Working RAM)

* **Mechanism:** An active Python dictionary array.
* **Function:** Holds the last 15 conversational turns. It allows Ada to understand immediate conversational context, resolve pronouns, and maintain natural flow.
* **Lifecycle:** Once the array exceeds 15 turns, the oldest chunks are evicted from VRAM and passed to Tier 3 for consolidation.

### Tier 2: Semantic Memory (The "Shadow Scribe")

* **Mechanism:** `user_profile.json` and `ada_profile.json`.
* **Function:** Stores immutable, permanent facts (e.g., user's name, pets, preferences, or Ada's internal state).
* **The Air-Gapped Engine:** A background thread analyzes the user's latest statement and extracts hard facts. It is explicitly "air-gapped" from Ada's dialogue to prevent cross-contamination or hallucinations.
* **LLM-Deduplication:** Before saving a new fact, the system injects the `KNOWN FACTS` into the prompt, forcing the LLM to verify that the new information is not a duplicate or a less-specific version of an existing memory.

### Tier 3: Episodic Memory (Medium-Term Journal)

* **Mechanism:** `timeline_log.json` powered by **Vector RAG** (Retrieval-Augmented Generation).
* **Function:** Acts as a chronological stream diary. Discarded turns from Tier 1 are compressed by a background LLM task into a single, dense chronological sentence (e.g., "User spent 45 minutes debugging the TTS audio queue.").
* **Semantic Search:** When the user speaks, an ultra-fast embedding model (`all-MiniLM-L6-v2`) converts their text into a mathematical vector and measures the Cosine Similarity against the entire timeline log. Relevant past memories are invisibly injected into Ada's context window before she replies.

### Tier 4: Deep Sleep Consolidation (Long-Term Archives)

* **Mechanism:** `archive_log.json` triggered by an offline batch script.
* **Function:** Mimics biological sleep cycles. Over weeks of use, the episodic journal becomes bloated.
* **Execution:** When triggered, the engine feeds hundreds of lines from the timeline log into Llama 3.1, instructing it to synthesize a dense, 3-sentence "Epoch" paragraph summarizing the major projects, themes, and milestones of that period.
* **The Wipe:** The newly synthesized Epoch is saved to deep storage, and the medium-term timeline log is wiped clean to restore lightning-fast search speeds.

---

## 4. The Autonomous Lifecycle (Voice Intercept)

The system does not require manual terminal commands to shut down or trigger memory consolidation. It utilizes a **Voice Command Interceptor** integrated directly into the transcription loop.

* When the microphone transcribes the exact phrase `good night ada` (ignoring punctuation or casing), the system intercepts the text before it reaches the LLM.
* It immediately queues a localized TTS goodbye message.
* It waits for the playback queue to finish draining, then synchronously triggers the **Deep Sleep Consolidation** (Tier 4) using the already-loaded Llama model.
* Once archiving is complete, the script safely drops the VRAM payload, flushes all audio buffers, and gracefully executes `sys.exit(0)`.