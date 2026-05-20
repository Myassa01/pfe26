"""Pipeline RAG principal — orchestration de toutes les couches.

Routage intelligent : aucune liste de mots-clés FR. Le schéma des sources est
découvert automatiquement au démarrage et un IntentRouter LLM classifie
chaque question pour décider du bypass / de la source / de la colonne.
"""
import hashlib
import logging
import os
import re
import threading
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

from .ingestion.loader import load_directory, Document
from .ingestion.chunker import chunk_documents
from .ingestion.embedder import Embedder
from .retrieval.vector_store import VectorStore
from .retrieval.bm25_search import BM25Search, BM25Document
from .reranking.reranker import CrossEncoderReranker
from .generation.llm import HFClient
from .generation.query_transform import QueryTransformer
from .generation.intent_router import IntentRouter, SchemaDiscovery
from .structured import StructuredQueryEngine
from .graph.graph import build_rag_graph


class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self._graph_lock = threading.Lock()

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

        self.structured = StructuredQueryEngine(config.docs_dir)

        self.schema = self._build_combined_schema(config.docs_dir)
        self.intent_router = IntentRouter(llm=self.llm, schema=self.schema)

        self._graph = self._build_graph()

        logger.info("=" * 50)
        logger.info("Pipeline prêt.")

    def _build_graph(self):
        return build_rag_graph({
            "query_transformer": self.query_transformer,
            "intent_router":     self.intent_router,
            "embedder":          self.embedder,
            "vector_store":      self.vector_store,
            "bm25":              self.bm25,
            "reranker":          self.reranker,
            "llm":               self.llm,
            "structured":        self.structured,
            "schema":            self.schema,
            "config":            self.config,
        })

    def _build_combined_schema(self, docs_dir: str) -> Dict[str, dict]:
        schema: Dict[str, dict] = {}

        for table_name, info in self.structured.schema().items():
            user_cols = self.structured.tables[table_name].get("user_columns") or info["columns"]
            schema[table_name] = {
                "columns": user_cols,
                "samples": self.structured.samples(table_name, max_per_col=3),
                "row_count": info.get("row_count", 0),
                "is_doc": False,
                "filename": info["filename"],
                "structured": True,
            }

        from pathlib import Path
        from .generation.intent_router import SchemaDiscovery
        doc_disc = SchemaDiscovery(docs_dir)
        for file in sorted(Path(docs_dir).rglob("*")) if Path(docs_dir).exists() else []:
            if not file.is_file():
                continue
            ext = file.suffix.lower()
            if ext in doc_disc.DOC_EXTS and ext not in doc_disc.EXCEL_EXTS:
                stem = doc_disc._normalize_stem(file.name)
                if stem not in schema:
                    schema[stem] = {
                        "columns": [], "samples": {}, "is_doc": True,
                        "filename": file.name, "structured": False,
                    }
        return schema

    @staticmethod
    def _normalize_stem(fname: str) -> str:
        stem = fname.rsplit(".", 1)[0] if "." in fname else fname
        stem = stem.upper().strip()
        stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
        stem = re.sub(r"\s*_\d+\s*$", "", stem)
        return stem.strip()

    # ── DÉDUPLICATION ────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_chunks(chunks: list) -> list:
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
        docs_dir = docs_dir or self.config.docs_dir

        logger.info("=" * 50)
        logger.info("INGESTION PIPELINE")
        logger.info("=" * 50)

        if reset:
            logger.info("Réinitialisation du vector store...")
            self.vector_store.reset()
            if os.path.exists(self.config.bm25_index_path):
                os.remove(self.config.bm25_index_path)

        logger.info("[1/4] Chargement des documents depuis '%s'...", docs_dir)
        documents = load_directory(docs_dir)
        if not documents:
            raise ValueError(f"Aucun document trouvé dans: {docs_dir}")
        logger.info("  → %d documents chargés", len(documents))

        logger.info("[2/4] Découpage (chunk_size=%d tokens, overlap=%d)...",
                    self.config.chunk_size, self.config.chunk_overlap)
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            embedding_model=self.config.embedding_model,
        )
        logger.info("  → %d chunks créés", len(chunks))
        chunks = self._deduplicate_chunks(chunks)

        logger.info("[3/4] Génération des embeddings (%s)...", self.config.embedding_model)
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress=True,
        )
        logger.info("  → Shape: %s", embeddings.shape)

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

        # ── Recharge DuckDB + schéma sans redémarrer l'app ──────────────────
        self.structured.reload(docs_dir)
        self.schema = self._build_combined_schema(docs_dir)
        self.intent_router = IntentRouter(llm=self.llm, schema=self.schema)
        with self._graph_lock:
            self._graph = self._build_graph()
        logger.info("DuckDB et schéma rechargés (%d table(s)).", len(self.structured.tables))

        logger.info("=" * 50)
        logger.info("Ingestion terminée: %d chunks indexés", len(chunks))
        logger.info("=" * 50)

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embedding_dim": int(embeddings.shape[1]),
        }

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
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

        # ── Recharge DuckDB + schéma sans redémarrer l'app ──────────────────
        self.structured.reload(self.config.docs_dir)
        self.schema = self._build_combined_schema(self.config.docs_dir)
        self.intent_router = IntentRouter(llm=self.llm, schema=self.schema)
        with self._graph_lock:
            self._graph = self._build_graph()

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embedding_dim": int(embeddings.shape[1]),
        }

    # ── QUERY ────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        use_query_transform: bool = False,
        stream: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        start = time.time()

        if use_query_transform:
            logger.warning("use_query_transform=True non supporté en mode LangGraph — ignoré.")

        with self._graph_lock:
            graph = self._graph
        final_state = graph.invoke({
            "question": question,
            "history":  history or [],
        })

        elapsed = round(time.time() - start, 2)

<<<<<<< HEAD
        resolved = final_state.get("resolved_question", question)
        return {
            "question":          question,
            "resolved_question": resolved if resolved != question else None,
            "search_query":      final_state.get("search_query", question),
            "answer":            final_state.get("answer", ""),
            "sources":           final_state.get("sources", []),
            "chunks_used":       final_state.get("chunks_used", 0),
            "elapsed_seconds":   elapsed,
            "intent":            final_state.get("intent_data", {}),
            "warnings":          final_state.get("warnings", []),
=======
        return {
            "question":        question,
            "search_query":    final_state.get("search_query", question),
            "answer":          final_state.get("answer", ""),
            "sources":         final_state.get("sources", []),
            "chunks_used":     final_state.get("chunks_used", 0),
            "elapsed_seconds": elapsed,
            "intent":          final_state.get("intent_data", {}),
            "warnings":        final_state.get("warnings", []),
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
        }
