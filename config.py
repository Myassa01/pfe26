from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ── LLM HuggingFace ─────────────────────────────────────────────────────
    # Colab gratuit (T4 15GB) → Qwen/Qwen2.5-1.5B-Instruct  (~3GB)
    # Colab Pro    (A100)     → Qwen/Qwen2.5-7B-Instruct    (~14GB)
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    # ── Embeddings multilingues ──────────────────────────────────────────────
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_batch_size: int = 32

    # ── Reranker ────────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "rag_documents"

    # ── BM25 ────────────────────────────────────────────────────────────────
    bm25_index_path: str = "./data/bm25_index.pkl"

    # ── Chunking (en tokens, pas en caractères) ────────────────────────────
    # Taille mesurée via le tokenizer du modèle d'embedding.
    # 256 tokens ≈ 600-800 caractères français — une fiche RH complète.
    chunk_size: int = 256
    chunk_overlap: int = 32

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_dense: int = 20   # ↑ augmenté pour ne pas rater des résultats
    top_k_sparse: int = 20  # ↑ augmenté (BM25 fort sur les noms propres)
    top_k_after_rerank: int = 5
    rrf_k: int = 60

    # ── Mode exhaustif (questions de type "liste-moi tout") ─────────────────
    # Seuil de pertinence du reranker : tout chunk au-dessus est inclus.
    # Cross-encoder mmarco : scores typiques entre -12 (hors sujet) et +12 (exact).
    # Un seuil de -3.0 garde les chunks "au moins vaguement pertinents".
    # (les scores mmarco sont souvent négatifs pour du contenu pertinent)
    rerank_min_score: float = -3.0
    # Nombre max de chunks à envoyer au LLM en mode exhaustif
    max_chunks_exhaustive: int = 20
    # max_tokens élevé pour les réponses longues (listes, tableaux)
    llm_max_tokens_long: int = 2048

    # ── Chemins ─────────────────────────────────────────────────────────────
    docs_dir: str = "./documents"
    data_dir: str = "./data"


config = Config()
