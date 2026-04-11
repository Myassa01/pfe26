from dataclasses import dataclass
 
 
@dataclass
class Config:
    # ── LLM via Ollama ──────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.0   # 0 = réponses plus déterministes/factuelles
    llm_max_tokens: int = 512      # Petit modèle → réponses courtes et précises
 
    # ── Embeddings multilingues (français supporté) ─────────────────
    # paraphrase-multilingual-MiniLM-L12-v2 : 118MB, supporte 50+ langues dont FR
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
 
    # ── Reranker multilingue ────────────────────────────────────────
    # ms-marco-MiniLM-L-6-v2 reste correct même en FR pour le reranking
    # car il score la pertinence sémantique générale
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
 
    # ── ChromaDB ────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "rag_documents"
 
    # ── BM25 ────────────────────────────────────────────────────────
    bm25_index_path: str = "./data/bm25_index.pkl"
 
    # ── Chunking ────────────────────────────────────────────────────
    # Chunks plus petits = contexte plus précis = meilleure réponse
    chunk_size: int = 500
    chunk_overlap: int = 50
 
    # ── Retrieval ───────────────────────────────────────────────────
    top_k_dense: int = 15
    top_k_sparse: int = 15
    top_k_after_rerank: int = 5   # 5 chunks max pour petit modèle (contexte limité)
    rrf_k: int = 60
 
    # ── Chemins ─────────────────────────────────────────────────────
    docs_dir: str = "./documents"
    data_dir: str = "./data"
 
 
config = Config()