"""Reranker local via Cross-Encoder (sentence-transformers)."""
from typing import List, Dict, Any


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        print(f"Chargement du reranker: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Réordonne les documents en calculant un score de pertinence
        query↔document via le cross-encoder.
        """
        if not documents:
            return []

        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.model.predict(pairs)

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, doc) in enumerate(scored[:top_k], 1):
            d = doc.copy()
            d["rerank_score"] = float(score)
            d["rank"] = rank
            results.append(d)

        return results
