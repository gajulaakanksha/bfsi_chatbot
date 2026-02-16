"""Tier 1 – Dataset Matcher.

Embeds the Alpaca dataset instructions using SentenceTransformers and
performs cosine-similarity search at query time.  If the best match
exceeds DATASET_MATCH_THRESHOLD the cached output is returned directly,
bypassing the SLM and RAG layers.
"""
import json
import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DATASET_PATH = os.path.join("data", "alpaca_bfsi_dataset.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
THRESHOLD = float(os.getenv("DATASET_MATCH_THRESHOLD", "0.85"))


class DatasetMatcher:
    """Find the closest pre-curated answer from the Alpaca dataset."""

    def __init__(self, dataset_path: str = DATASET_PATH, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        # Pre-compute instruction embeddings
        instructions = [s["instruction"] for s in self.dataset]
        self.instruction_embeddings = self.model.encode(
            instructions, normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, threshold: float = THRESHOLD):
        """Return (answer, score) if above threshold, else (None, score)."""
        query_emb = self.model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.instruction_embeddings, query_emb.T).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= threshold:
            return self.dataset[best_idx]["output"], best_score
        return None, best_score
