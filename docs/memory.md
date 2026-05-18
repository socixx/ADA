# Ada Core Memory Architecture: Technical Specification

## Overview

The Ada Core Memory Architecture is a biologically-inspired, multi-tiered AI memory system designed specifically for local, persistent voice companions running on consumer hardware (e.g., RTX 3090). It solves the core problem of infinite context bloat by dividing memories into distinct physical spaces and lifespans, ensuring zero latency degradation during long, 6-hour streaming sessions.

Instead of a monolithic "flat file" approach, Ada's brain categorizes memories into **Short-Term (Working RAM)**, **Semantic Long-Term (Profiles)**, **Episodic Medium-Term (Journals)**, and **Deep Storage (Epoch Archives)**.

---

## 1. Tier 1: Short-Term Memory (Working RAM)

Short-Term Memory is responsible for immediate conversational coherence. It allows Ada to understand pronoun resolution ("she", "it") and track the immediate flow of the conversation.

* **Mechanism:** An active Python array (`self.history`).
* **Capacity:** Strictly capped at 15 conversational turns.
* **Behavior:** Once the 15-turn threshold is crossed, the oldest 4 turns are evicted from active VRAM. This guarantees that the Time-To-First-Token (TTFT) remains instant, regardless of how long the user has been talking.
* **Pipeline:** Evicted turns are never deleted; they are passed to the Tier 3 background worker for consolidation.

---

## 2. Tier 2: Semantic Memory (The "Air-Gapped" Scribe)

Semantic Memory holds immutable, permanent facts about both the User and Ada. This ensures Ada always knows who she is talking to and remembers her own internal state and promises.

* **Mechanism:** Two distinct JSON files: `user_profile.json` and `ada_profile.json`.
* **The Scribe Engine:** A background thread running asynchronously during the TTS playback. It analyzes the user's latest statement and extracts hard facts (e.g., "Evan has a dog named Bailey").
* **Air-Gapped Design:** To prevent the 8B LLM from hallucinating or cross-contaminating data, the Scribe is strictly "air-gapped." It is blinded to Ada's replies and only evaluates the user's raw text. 
* **LLM-Driven Deduplication:** Instead of using rigid Python string matching, the prompt injects the `KNOWN FACTS` into the Scribe's context. The LLM natively compares the new statement against existing facts to prevent redundant or generic overlaps (e.g., rejecting "Evan has a pet" if "Evan has a dog" is already known).
* **Self-Evolving Identity:** The system runs a dynamic regex scanner over the user profile. When it detects a name, it automatically pivots its internal identity variables, allowing it to seamlessly handle multi-speaker environments in the future.

---

## 3. Tier 3: Episodic Memory (Vector RAG)

Episodic Memory serves as a chronological stream diary. It remembers *what happened* and *when*, providing the context needed for Ada to recall past coding sessions, jokes, or project milestones.

* **Mechanism:** `timeline_log.json` powered by a local, ultra-fast embedding model (`all-MiniLM-L6-v2`).
* **Consolidation Pipeline:** When Tier 1 (Short-Term) drops the oldest 4 turns, a background worker compresses those raw chat lines into a single, dense chronological sentence (e.g., `"2026-05-18 10:15 - Evan debugged the semantic VAD confidence gate."`).
* **Semantic Search (RAG):** When the user speaks, their audio is instantly converted into a vector. The system calculates the Cosine Similarity against every entry in the timeline journal. 
* **Injection:** If a strong semantic match is found (Score > 0.35), that specific diary entry is retrieved and invisibly injected into Ada's system prompt *before* she replies. This creates the illusion of organic, unprompted recall.

---

## 4. Tier 4: Deep Sleep Consolidation (Epoch Archives)

As the Tier 3 episodic journal grows over weeks or months, it risks becoming bloated. The **Deep Sleep Cycle** mimics human cognitive consolidation, turning day-to-day journals into broad, permanent wisdom.

* **Mechanism:** `archive_log.json` triggered by a voice-activated interceptor.
* **The Voice Hook:** If the system hears `"good night ada"`, it intercepts the text, bypasses standard LLM generation, speaks a goodbye message, and locks the thread to begin consolidation.
* **Compression:** The Deep Sleep engine ingests the entire `timeline_log.json` and instructs the LLM to write a dense, 3-sentence narrative paragraph summarizing the core themes, projects, and milestones of the recent sessions (The "Epoch").
* **The Wipe:** The Epoch is appended to `archive_log.json`, and the `timeline_log.json` is wiped completely clean (saving only the last 3 entries for continuity upon waking). 
* **Global Search:** The Vector RAG engine simultaneously searches both the active timeline and the Deep Storage archives, ensuring Ada never loses access to long-term memories.

---

## Conclusion

By isolating memory into specific functional tiers, Ada achieves the conversational depth of a massive enterprise agent while operating efficiently within the strict hardware constraints of a single local GPU. The system separates the "fast thinking" required for live voice interaction from the "slow thinking" required for identity management and memory consolidation.