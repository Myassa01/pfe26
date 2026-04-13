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

    # ── Chunking ─────────────────────────────────────────────────────────────
    # ↑ chunk_size augmenté : les fiches individuelles font ~300 chars
    # → un chunk = une fiche complète = meilleur contexte
    chunk_size: int = 600
    chunk_overlap: int = 50

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_dense: int = 20   # ↑ augmenté pour ne pas rater des résultats
    top_k_sparse: int = 20  # ↑ augmenté (BM25 fort sur les noms propres)
    top_k_after_rerank: int = 5
    rrf_k: int = 60

    # ── Chemins ─────────────────────────────────────────────────────────────
    docs_dir: str = "./documents"
    data_dir: str = "./data"

    ollama_base_url: str = "http://localhost:11434"


config = Config()
