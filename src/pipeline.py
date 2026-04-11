"""Pipeline RAG principal — orchestration de toutes les couches."""
import os
import time
from typing import List, Dict, Any, Optional

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

        print("\nInitialisation du pipeline RAG local...")
        print("=" * 50)

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
        print("=" * 50)
        print("Pipeline prêt.\n")

    # ── INGESTION ────────────────────────────────────────────────────────────

    def ingest(self, docs_dir: Optional[str] = None, reset: bool = False) -> Dict[str, Any]:
        """Ingère les documents du dossier dans le RAG."""
        docs_dir = docs_dir or self.config.docs_dir

        print(f"\n{'='*50}")
        print("INGESTION PIPELINE")
        print(f"{'='*50}")

        if reset:
            print("\nRéinitialisation du vector store...")
            self.vector_store.reset()
            if os.path.exists(self.config.bm25_index_path):
                os.remove(self.config.bm25_index_path)

        # 1. Chargement
        print(f"\n[1/4] Chargement des documents depuis '{docs_dir}'...")
        documents = load_directory(docs_dir)
        if not documents:
            raise ValueError(f"Aucun document trouvé dans: {docs_dir}")
        print(f"  → {len(documents)} documents chargés")

        # 2. Chunking
        print(f"\n[2/4] Découpage (chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap})...")
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        print(f"  → {len(chunks)} chunks créés")

        # 3. Embeddings
        print(f"\n[3/4] Génération des embeddings ({self.config.embedding_model})...")
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress=True,
        )
        print(f"  → Shape: {embeddings.shape}")

        # 4. Indexation
        print("\n[4/4] Indexation (ChromaDB + BM25)...")
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

        print(f"\n{'='*50}")
        print(f"Ingestion terminée: {len(chunks)} chunks indexés")
        print(f"{'='*50}\n")

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
        )
        if not chunks:
            return {"documents": len(documents), "chunks": 0, "embedding_dim": 0}

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
            print("  [1/4] Transformation de la requête...")
            search_query = self.query_transformer.rewrite(question)
            if search_query != question:
                print(f"        → {search_query}")
        else:
            search_query = question

        # Étape 2 : Recherche hybride
        print("  [2/4] Recherche hybride...")
        query_emb = self.embedder.embed_single(search_query)
        dense  = self.vector_store.search(query_emb, k=self.config.top_k_dense)
        sparse = self.bm25.search(search_query, k=self.config.top_k_sparse)
        hybrid = reciprocal_rank_fusion(dense, sparse, k=self.config.rrf_k)
        print(f"        Dense: {len(dense)} | Sparse: {len(sparse)} | RRF: {len(hybrid)}")

        # Étape 3 : Reranking
        print("  [3/4] Reranking...")
        reranked = self.reranker.rerank(
            query=search_query,
            documents=hybrid[:20],
            top_k=self.config.top_k_after_rerank,
        )
        print(f"        → {len(reranked)} chunks retenus")

        # Étape 4 : Génération
        print("  [4/4] Génération LLM...")
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