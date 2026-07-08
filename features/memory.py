import time
import json
import os
import threading
import re
import numpy as np
import config
from fastembed import TextEmbedding
from openai import OpenAI

class MemoryManager:
    def __init__(self, model_lock):
        self.model_lock = model_lock
        
        # Connect to the local vLLM Docker node for memory summarization tasks
        self.client = OpenAI(base_url=f"http://localhost:{config.LLM_PORT}/v1", api_key="token")
        
        print("[Memory] Loading FastEmbed ONNX Engine ('all-MiniLM-L6-v2')...")
        self.embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.timeline_embeddings = None
        
        # --- CREATE MEMORY DIRECTORY ---
        self.memory_dir = os.path.join(config.BASE_DIR, "memory")
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)
            
        self.profile_file = os.path.join(self.memory_dir, "user_profile.json")     
        self.timeline_file = os.path.join(self.memory_dir, "timeline_log.json")   
        self.ada_file = os.path.join(self.memory_dir, "ada_profile.json")       
        self.archive_file = os.path.join(self.memory_dir, "archive_log.json")
        
        for file in [self.profile_file, self.timeline_file, self.ada_file, self.archive_file]:
            if not os.path.exists(file):
                with open(file, "w") as f:
                    json.dump([], f)
                    
        self.user_name = "User" 
        self._refresh_identity() 

    def _refresh_identity(self):
        try:
            with open(self.profile_file, "r") as f:
                profile = json.load(f)
                for fact in profile:
                    match = re.search(r"(?:user'?s name is|user is named)\s+([A-Za-z]+)", fact, re.IGNORECASE)
                    if match and match.group(1).lower() != "user":
                        self.user_name = match.group(1).capitalize()
                        return
        except Exception:
            pass
        self.user_name = "User" 

    def _cosine_similarity(self, query_vec, corpus_vecs):
        """Native numpy cosine similarity to bypass heavy dependencies."""
        dot_product = np.dot(corpus_vecs, query_vec)
        norm_query = np.linalg.norm(query_vec)
        norm_corpus = np.linalg.norm(corpus_vecs, axis=1)
        return dot_product / (norm_corpus * norm_query)

    def load_memories(self, current_user_text):
        self._refresh_identity() 
        memory_context = ""
        
        try:
            with open(self.profile_file, "r") as f:
                profile = json.load(f)
                if profile:
                    memory_context += f"[USER BIOGRAPHY - {self.user_name.upper()}]:\n" + "\n".join(f"- {p}" for p in profile) + "\n\n"
            
            with open(self.ada_file, "r") as f:
                ada_state = json.load(f)
                if ada_state:
                    memory_context += "[ADA'S INTERNAL STATE]:\n" + "\n".join(f"- {p}" for p in ada_state) + "\n\n"
        except Exception as e:
            print(f"[Profile Load Error] {e}")

        try:
            with open(self.timeline_file, "r") as f:
                timeline = json.load(f)
                
            archive = []
            if os.path.exists(self.archive_file):
                with open(self.archive_file, "r") as f:
                    archive = json.load(f)
            
            searchable_database = archive + timeline
                
            if searchable_database:
                # FastEmbed returns a generator, so we convert to a numpy array
                if self.timeline_embeddings is None or len(self.timeline_embeddings) != len(searchable_database):
                    embeddings = list(self.embedder.embed(searchable_database))
                    self.timeline_embeddings = np.array(embeddings)
                
                query_embedding = list(self.embedder.embed([current_user_text]))[0]
                
                # Perform native numpy semantic search
                scores = self._cosine_similarity(query_embedding, self.timeline_embeddings)
                
                # Get top 3 indices sorted by score
                top_k_indices = np.argsort(scores)[-3:][::-1]
                
                relevant_logs = []
                for idx in top_k_indices:
                    if scores[idx] >= 0.35:
                        relevant_logs.append(searchable_database[idx])
                
                if relevant_logs:
                    memory_context += "[RECALLED STREAM MEMORIES]:\n" + "\n".join(f"- {log}" for log in relevant_logs) + "\n\n"
                    
        except Exception as e:
            print(f"[RAG Retrieval Error] {e}")
            
        return memory_context if memory_context else "[No historical data established yet.]\n\n"

    def consolidate_to_timeline(self, discarded_turns):
        def consolidation_task():
            with self.model_lock:
                self._refresh_identity()
                log_str = ""
                for turn in discarded_turns:
                    role = "Ada" if turn["role"] == "assistant" else self.user_name
                    log_str += f"{role}: {turn['content']}\n"
                
                prompt = (
                    "You are a history consolidation engine. Review these older lines from a live stream chat "
                    f"and summarize what {self.user_name} and Ada did or talked about into ONE short chronological sentence.\n"
                    "Focus strictly on actions, technical achievements, or topics introduced. Do not include greetings.\n\n"
                    f"CHAT BLOCK:\n{log_str}\n"
                    "SUMMARY SENTENCE (Start with an action verb, max 12 words):"
                )
                
                try:
                    response = self.client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=40
                    )
                    summary = response.choices[0].message.content.strip()
                    
                    if summary and "NONE" not in summary.upper():
                        print(f"[Long-Term Timeline] Consolidated: \"{summary}\"")
                        with open(self.timeline_file, "r") as f:
                            timeline = json.load(f)
                        timeline.append(f"{time.strftime('%Y-%m-%d %H:%M')} - {summary}")
                        with open(self.timeline_file, "w") as f:
                            json.dump(timeline, f, indent=4)
                except Exception as e:
                    print(f"[Timeline Write Error] {e}")

        threading.Thread(target=consolidation_task, daemon=True).start()

    def shadow_scribe_worker(self, current_user_text, history):
        def extraction_task():
            with self.model_lock:
                self._refresh_identity()
                try:
                    with open(self.profile_file, "r") as f:
                        user_memories = json.load(f)
                except Exception:
                    user_memories = []
                    
                existing_facts_str = "\n".join(f"- {f}" for f in user_memories) if user_memories else "None."

                extraction_prompt = (
                    f"You are a strict data filter. Compare {self.user_name}'s LATEST STATEMENT against the KNOWN FACTS.\n"
                    f"Extract a fact ONLY if it is a permanent personal detail that is ENTIRELY NEW and NOT covered by the known facts.\n\n"
                    f"KNOWN FACTS:\n{existing_facts_str}\n\n"
                    "RULES:\n"
                    f"1. Output format: 'USER FACT: {self.user_name} [fact]'\n"
                    "2. If the information is already logically covered in the KNOWN FACTS, output EXACTLY the word 'NONE' and NOTHING ELSE.\n"
                    "3. If the statement is conversational filler, output EXACTLY the word 'NONE' and NOTHING ELSE.\n"
                    "4. DO NOT extract temporary feelings, actions, or questions.\n\n"
                    f"LATEST STATEMENT: \"{current_user_text}\"\n"
                    "OUTPUT:"
                )
                
                try:
                    response = self.client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=[{"role": "user", "content": extraction_prompt}],
                        temperature=0.1,
                        max_tokens=40
                    )
                    result = response.choices[0].message.content.strip()
                    
                    for line in result.split("\n"):
                        clean_line = line.strip()
                        if "USER FACT:" in clean_line:
                            fact = clean_line.split("USER FACT:")[-1].strip()
                            if fact and "NONE" not in fact.upper():
                                print(f"[Profile Memory] Saved User Fact: \"{fact}\"")
                                user_memories.append(fact)
                                with open(self.profile_file, "w") as f:
                                    json.dump(user_memories, f, indent=4)
                                self._refresh_identity()
                except Exception as e:
                    print(f"[Shadow Scribe Error] {e}")

        threading.Thread(target=extraction_task, daemon=True).start()

    def deep_sleep_consolidation(self):
        print("\n[Memory] Initiating Deep Sleep Memory Consolidation...")
        with self.model_lock:
            try:
                with open(self.timeline_file, "r") as f:
                    timeline = json.load(f)
            except Exception:
                timeline = []

            if len(timeline) < 5:
                print("[Memory] Not enough memories to consolidate. Ada can keep sleeping.")
                return
            
            timeline_str = "\n".join(f"- {entry}" for entry in timeline)
            current_date = time.strftime('%B %d, %Y')

            prompt = (
                "You are a cognitive consolidation engine. Read the following chronological diary of a user's recent stream sessions.\n"
                "Your job is to compress this entire log into a SINGLE, dense, 3-sentence paragraph summarizing the core technical themes, "
                "major projects worked on, and any significant life events.\n\n"
                "RULES:\n"
                "1. Write in the third person past-tense.\n"
                "2. DO NOT output a list or bullet points. Output exactly one cohesive paragraph.\n"
                f"DIARY LOG:\n{timeline_str}\n\n"
                "EPOCH SUMMARY PARAGRAPH:"
            )

            try:
                response = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=150
                )
                summary = response.choices[0].message.content.strip()
                
                if summary:
                    with open(self.archive_file, "r") as f:
                        archive = json.load(f)
                    
                    archive_entry = f"[{current_date} Epoch]: {summary}"
                    archive.append(archive_entry)
                    
                    with open(self.archive_file, "w") as f:
                        json.dump(archive, f, indent=4)
                        
                    with open(self.timeline_file, "w") as f:
                        json.dump(timeline[-3:], f, indent=4)
                        
                    print(f"[Memory] Consolidation Complete:\n{archive_entry}\n")
            except Exception as e:
                print(f"[Archive Write Error] {e}")