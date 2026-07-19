import os
import numpy as np
from fastembed import TextEmbedding

class SemanticEvaluator:
    def __init__(self):
        self.model = None

    def _initialize_model(self):
        """Initializes the fastembed pipeline with the large BGE model and CUDA binding."""
        print("[Semantic Judge] Binding large evaluator engine to host ONNX framework...")
        
        # Explicitly request the large 1024-dimension model and bind to the GPU
        self.model = TextEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            providers=["CUDAExecutionProvider"]
        )
        print("[Semantic Judge] Evaluator online via CUDA.")

    def evaluate_batch(self, benchmark_results):
        """
        Processes completed benchmark metrics, computes the cosine similarity 
        against your expected ground-truth targets, and scales the score from 0.0 to 10.0.
        """
        if self.model is None:
            self._initialize_model()
            
        print(f"[Semantic Judge] Commencing fastembed evaluation loop for {len(benchmark_results)} runs...")

        for run in benchmark_results:
            target_mock = run.get("target_mock", "")
            generated_text = run.get("generated_text", "")
            
            if not target_mock.strip() or not generated_text.strip():
                run["target_convergence_accuracy"] = 0.0
                continue
            
            # fastembed returns generators; convert them to raw list embeddings
            # Note: We pass strings wrapped in a list as expected by the fastembed API
            embeddings = list(self.model.embed([target_mock, generated_text]))
            
            vec_target = embeddings[0]
            vec_actual = embeddings[1]
            
            # Compute Cosine Similarity: (A • B) / (||A|| * ||B||)
            dot_product = np.dot(vec_target, vec_actual)
            norm_target = np.linalg.norm(vec_target)
            norm_actual = np.linalg.norm(vec_actual)
            
            similarity = dot_product / (norm_target * norm_actual) if (norm_target * norm_actual) > 0 else 0.0
            
            # Scale the similarity spectrum (-1.0 to 1.0) onto a 0.0 to 10.0 scale
            calibrated_score = (similarity + 1) * 5.0
            
            # Failsafe: If target substring explicitly matches, ensure it meets baseline pass thresholds
            if target_mock.lower() in generated_text.lower():
                calibrated_score = max(calibrated_score, 9.5)
                
            run["target_convergence_accuracy"] = float(round(min(max(calibrated_score, 0.0), 10.0), 1))
            print(f"  -> Evaluated Run {run.get('preset_id', 'Unknown')}: {run['target_convergence_accuracy']}/10.0")
            
        return benchmark_results

# Singleton instance
judge = SemanticEvaluator()