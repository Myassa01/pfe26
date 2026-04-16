"""Pipeline RAG principal — orchestration de toutes les couches."""
import hashlib
import logging
import os
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

from .ingestion.loader import load_directory, Document
from .ingestion.chunker import chunk_documents
from .ingestion.embedder import Embedder
from .retrieval.vector_store import VectorStore
from .retrieval.bm25_search import BM25Search, BM25Document
from .retrieval.hybrid_search import reciprocal_rank_fusion
from .reranking.reranker import CrossEncoderReranker
from .generation.llm import HFClient
from .generation.query_transform import QueryTransformer

# ── Prompts optimisés pour petit modèle (qwen2.5:0.5b) ──────────────────────
# Règle d'or : prompt court + instruction simple = meilleure réponse

_SYSTEM_PROMPT = """Tu es un assistant RH. Réponds en français uniquement à partir du contexte fourni.
Si l'information n'est pas dans le contexte, réponds : "Je ne trouve pas cette information dans les documents."
Cite le fichier source entre crochets."""

_GENERATION_PROMPT = """Contexte:
{context}

{history}Question: {question}

Réponse:"""


class RAGPipeline:
    def __init__(self, config):
        self.config = config

        logger.info("Initialisation du pipeline RAG local...")
        logger.info("=" * 50)

        self.embedder = Embedder(
            model_name=config.embedding_model,
            device=config.embedding_device,
        )

        self.vector_store = VectorStore(
            persist_dir=config.chroma_persist_dir,
            collection_name=config.collection_name,
        )

        self.bm25 = BM25Search()
        self.bm25.load(config.bm25_index_path)

        self.reranker = CrossEncoderReranker(model_name=config.reranker_model)

        self.llm = HFClient(
            model=config.llm_model,
        )

        self.query_transformer = QueryTransformer(llm=self.llm)

        # Cache LRU pour les résultats de retrieval (évite de refaire embedding+search+rerank)
        self._retrieval_cache: Dict[str, List[Dict]] = {}
        self._cache_max_size: int = 128

        logger.info("=" * 50)
        logger.info("Pipeline prêt.")

    # ── DÉDUPLICATION ────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_chunks(chunks: list) -> list:
        """Supprime les chunks avec un contenu identique (hash SHA256)."""
        seen = set()
        unique = []
        for chunk in chunks:
            h = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(chunk)
        removed = len(chunks) - len(unique)
        if removed > 0:
            logger.info("  → %d chunk(s) dupliqué(s) supprimé(s)", removed)
        return unique

    # ── INGESTION ────────────────────────────────────────────────────────────

    def ingest(self, docs_dir: Optional[str] = None, reset: bool = False) -> Dict[str, Any]:
        """Ingère les documents du dossier dans le RAG."""
        docs_dir = docs_dir or self.config.docs_dir

        logger.info("=" * 50)
        logger.info("INGESTION PIPELINE")
        logger.info("=" * 50)

        if reset:
            logger.info("Réinitialisation du vector store...")
            self.vector_store.reset()
            if os.path.exists(self.config.bm25_index_path):
                os.remove(self.config.bm25_index_path)

        # 1. Chargement
        logger.info("[1/4] Chargement des documents depuis '%s'...", docs_dir)
        documents = load_directory(docs_dir)
        if not documents:
            raise ValueError(f"Aucun document trouvé dans: {docs_dir}")
        logger.info("  → %d documents chargés", len(documents))

        # 2. Chunking
        logger.info("[2/4] Découpage (chunk_size=%d tokens, overlap=%d)...", self.config.chunk_size, self.config.chunk_overlap)
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            embedding_model=self.config.embedding_model,
        )
        logger.info("  → %d chunks créés", len(chunks))
        chunks = self._deduplicate_chunks(chunks)

        # 3. Embeddings
        logger.info("[3/4] Génération des embeddings (%s)...", self.config.embedding_model)
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress=True,
        )
        logger.info("  → Shape: %s", embeddings.shape)

        # 4. Indexation
        logger.info("[4/4] Indexation (ChromaDB + BM25)...")
        self.vector_store.add(
            ids=[c.id for c in chunks],
            embeddings=list(embeddings),
            texts=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

        bm25_docs = [
            BM25Document(id=c.id, content=c.content, metadata=c.metadata)
            for c in chunks
        ]
        self.bm25.add_documents(bm25_docs)
        self.bm25.save(self.config.bm25_index_path)

        self._retrieval_cache.clear()

        logger.info("=" * 50)
        logger.info("Ingestion terminée: %d chunks indexés", len(chunks))
        logger.info("=" * 50)

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embedding_dim": int(embeddings.shape[1]),
        }

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Ingère une liste de Documents directement (sans lire un dossier)."""
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            embedding_model=self.config.embedding_model,
        )
        if not chunks:
            return {"documents": len(documents), "chunks": 0, "embedding_dim": 0}
        chunks = self._deduplicate_chunks(chunks)

        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress=False,
        )
        self.vector_store.add(
            ids=[c.id for c in chunks],
            embeddings=list(embeddings),
            texts=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        bm25_docs = [
            BM25Document(id=c.id, content=c.content, metadata=c.metadata)
            for c in chunks
        ]
        self.bm25.add_documents(bm25_docs)
        self.bm25.save(self.config.bm25_index_path)
        self._retrieval_cache.clear()

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embedding_dim": int(embeddings.shape[1]),
        }

    # ── QUERY ────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        use_query_transform: bool = False,   # Désactivé par défaut pour petits modèles
        stream: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Interroge le RAG et retourne la réponse avec ses sources."""
        start = time.time()

        # Étape 1 : Transformation de la requête (optionnelle)
        if use_query_transform:
            logger.info("  [1/4] Transformation de la requête...")
            search_query = self.query_transformer.rewrite(question)
            if search_query != question:
                logger.info("        → %s", search_query)
        else:
            search_query = question

        # Étape 2+3 : Recherche hybride + Reranking (avec cache)
        cache_key = search_query.strip().lower()
        if cache_key in self._retrieval_cache:
            logger.info("  [2/4] Recherche hybride... (cache hit)")
            logger.info("  [3/4] Reranking... (cache hit)")
            reranked = self._retrieval_cache[cache_key]
        else:
            logger.info("  [2/4] Recherche hybride...")
            query_emb = self.embedder.embed_single(search_query)
            dense  = self.vector_store.search(query_emb, k=self.config.top_k_dense)
            sparse = self.bm25.search(search_query, k=self.config.top_k_sparse)
            hybrid = reciprocal_rank_fusion(dense, sparse, k=self.config.rrf_k)
            logger.info("        Dense: %d | Sparse: %d | RRF: %d", len(dense), len(sparse), len(hybrid))

            logger.info("  [3/4] Reranking...")
            reranked = self.reranker.rerank(
                query=search_query,
                documents=hybrid[:20],
                top_k=self.config.top_k_after_rerank,
            )
            # Mise en cache (LRU simple)
            if len(self._retrieval_cache) >= self._cache_max_size:
                oldest_key = next(iter(self._retrieval_cache))
                del self._retrieval_cache[oldest_key]
            self._retrieval_cache[cache_key] = reranked
        logger.info("        → %d chunks retenus", len(reranked))

        # Étape 4 : Génération
        logger.info("  [4/4] Génération LLM...")
        context      = self._format_context(reranked)
        history_text = self._format_history(history) if history else ""
        prompt = _GENERATION_PROMPT.format(
            context=context,
            question=question,
            history=history_text,
        )

        if stream:
            answer_parts = []
            for token in self.llm.generate_stream(prompt=prompt, system=_SYSTEM_PROMPT):
                print(token, end="", flush=True)
                answer_parts.append(token)
            print()
            answer = "".join(answer_parts)
        else:
            answer = self.llm.generate(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )

        elapsed = round(time.time() - start, 2)

        return {
            "question":        question,
            "search_query":    search_query,
            "answer":          answer,
            "sources":         self._extract_sources(reranked),
            "chunks_used":     len(reranked),
            "elapsed_seconds": elapsed,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = []
        # Garde seulement les 4 derniers échanges pour ne pas surcharger le contexte
        for msg in history[-4:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "Historique:\n" + "\n".join(lines) + "\n\n"

    def _format_context(self, chunks: List[Dict]) -> str:
        """Formate le contexte de façon compacte pour les petits modèles."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            fname = chunk["metadata"].get("filename", "inconnu")
            # Format compact : juste le nom du fichier et le contenu
            parts.append(f"[{fname}]\n{chunk['content']}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: List[Dict]) -> List[str]:
        seen: set = set()
        sources = []
        for chunk in chunks:
            src = chunk["metadata"].get("filename", "inconnu")
            if src not in seen:
                sources.append(src)
                seen.add(src)
        return sources