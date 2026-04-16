"""Reranker local via Cross-Encoder (sentence-transformers)."""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        logger.info("Chargement du reranker: %s...", model_name)
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: Optional[float] = None,
        max_chunks: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Réordonne les documents via le cross-encoder.

        Si min_score est défini, retourne TOUS les documents au-dessus du seuil
        (plafonné à max_chunks). Sinon, retourne les top_k meilleurs.
        Cela permet aux questions de type "liste-moi tout" de recevoir
        plus de contexte que les questions ciblées.
        """
        if not documents:
            return []

        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.model.predict(pairs)

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, doc) in enumerate(scored, 1):
            if min_score is not None:
                # Mode seuil : garder tout ce qui est pertinent, jusqu'à max_chunks
                if float(score) < min_score:
                    break
                if rank > max_chunks:
                    break
            else:
                # Mode classique top-k
                if rank > top_k:
                    break

            d = doc.copy()
            d["rerank_score"] = float(score)
            d["rank"] = rank
            results.append(d)

        return results
