import os
from dataclasses import dataclass

try:
    import torch
    _cuda = torch.cuda.is_available()
except ImportError:
    _cuda = False

# ── Variables globales (importées par api.py, auth.py) ────────────────────────
DB_PATH        = "./data/organisation.db"
USERS_DB       = "./data/users.db"
CHROMA_PATH    = "./data/chroma_db"
DOCUMENTS_PATH = "./documents"
LLM_MODEL      = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
EMBED_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K          = 5


@dataclass
class Config:
    # ── LLM HuggingFace ───────────────────────────────────────────────────────
    llm_model:            str   = LLM_MODEL
    llm_temperature:      float = 0.0
    llm_max_tokens:       int   = 512
    llm_max_tokens_long:  int   = 1024

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model:      str   = EMBED_MODEL
    embedding_device:     str   = "cuda" if _cuda else "cpu"
    embedding_batch_size: int   = 32

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_model:       str   = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir:   str   = CHROMA_PATH
    collection_name:      str   = "rag_documents"

    # ── BM25 — IMPORTANT : .pkl pas .json ────────────────────────────────────
    bm25_index_path:      str   = "./data/bm25_index.pkl"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size:           int   = 256
    chunk_overlap:        int   = 32

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_dense:          int   = 20
    top_k_sparse:         int   = 20
    top_k_after_rerank:   int   = 5
    rrf_k:                int   = 60

    # ── Mode exhaustif ────────────────────────────────────────────────────────
    max_chunks_exhaustive: int  = 200

    # ── Validation LLM ────────────────────────────────────────────────────────
    validation_enabled:   bool  = False   # ← désactivé : trop lent sur petit modèle
    validation_batch_size: int  = 10

    # ── Chemins ───────────────────────────────────────────────────────────────
    docs_dir:  str = DOCUMENTS_PATH
    data_dir:  str = "./data"


config = Config()

# Crée les dossiers au démarrage
os.makedirs("./data", exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)
