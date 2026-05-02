"""Pipeline RAG principal — orchestration de toutes les couches.

Routage intelligent : aucune liste de mots-clés FR. Le schéma des sources est
découvert automatiquement au démarrage et un IntentRouter LLM classifie
chaque question pour décider du bypass / de la source / de la colonne.
"""
import hashlib
import logging
import os
import re
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
from .generation.intent_router import IntentRouter, SchemaDiscovery
from .structured import StructuredQueryEngine

# ── Prompts optimisés pour petit modèle (qwen2.5:0.5b) ──────────────────────

_SYSTEM_PROMPT = """Tu es un assistant RH. Réponds en français uniquement à partir du contexte fourni.
Chaque entrée du contexte est préfixée par sa source entre crochets (ex: [DIRECTION], [DEPARTEMENT], [SERVICE], [POSTE]).
Utilise UNIQUEMENT les données de la source pertinente à la question. Ignore les entrées provenant de sources non pertinentes.
Si l'information n'est pas dans le contexte, réponds : "Je ne trouve pas cette information dans les documents."
N'invente JAMAIS de données. Cite uniquement ce qui apparaît explicitement dans le contexte."""

_GENERATION_PROMPT = """Contexte:
{context}

{history}Question: {question}

Réponse:"""

_GENERATION_PROMPT_LIST = """Contexte:
{context}

{history}Question: {question}

IMPORTANT: La question demande une liste. Tu dois citer TOUS les éléments présents dans le contexte, sans en omettre.
Formatte la réponse sous forme de liste numérotée. Ne résume pas, ne regroupe pas, liste chaque élément.

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

        # Moteur de requêtage structuré (DuckDB) pour les fichiers Excel.
        # Toute question dont l'intent cible un fichier tabulaire bypasse le RAG
        # et passe par du SQL → 100% précis, déterministe, sans hallucination.
        self.structured = StructuredQueryEngine(config.docs_dir)

        # Découverte du schéma : on combine les tables structurées (DuckDB) +
        # les documents non-structurés (.docx, .pdf) pour le router d'intent.
        self.schema = self._build_combined_schema(config.docs_dir)
        self.intent_router = IntentRouter(llm=self.llm, schema=self.schema)

        # Cache LRU pour le retrieval
        self._retrieval_cache: Dict[tuple, List[Dict]] = {}
        self._cache_max_size: int = 128

        logger.info("=" * 50)
        logger.info("Pipeline prêt.")

    def _build_combined_schema(self, docs_dir: str) -> Dict[str, dict]:
        """Combine le schéma structuré (DuckDB) avec les documents non-structurés.
        Le résultat est passé à l'IntentRouter pour qu'il connaisse toutes les sources."""
        schema: Dict[str, dict] = {}

        # 1. Tables structurées : on enrichit avec des échantillons (utile pour le LLM)
        for table_name, info in self.structured.schema().items():
            schema[table_name] = {
                "columns": info["columns"],
                "samples": self.structured.samples(table_name),
                "is_doc": False,
                "filename": info["filename"],
                "structured": True,
            }

        # 2. Documents non-structurés : on délègue à SchemaDiscovery (gère .docx/.pdf)
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
        """SERVICE (1).xlsx → SERVICE"""
        stem = fname.rsplit(".", 1)[0] if "." in fname else fname
        stem = stem.upper().strip()
        stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
        stem = re.sub(r"\s*_\d+\s*$", "", stem)
        return stem.strip()

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

    # ── FILTRAGE PAR SOURCE ──────────────────────────────────────────────────

    def _filter_by_source(self, chunks: list, source: Optional[str]) -> list:
        """Privilégie les chunks issus de la source détectée par l'IntentRouter.
        Garde tous les chunks de la source + max 3 chunks d'autres sources."""
        if not source:
            return chunks
        from_relevant, from_other = [], []
        for chunk in chunks:
            fname = chunk["metadata"].get("filename", "")
            if self._normalize_stem(fname) == source:
                from_relevant.append(chunk)
            else:
                from_other.append(chunk)
        if not from_relevant:
            return chunks  # aucune correspondance → ne pas filtrer
        result = from_relevant + from_other[:3]
        logger.info("  → Filtre source: %d/%d chunks retenus (source: %s)",
                    len(result), len(chunks), source)
        return result

    # ── EXTRACTION DIRECTE GÉNÉRIQUE (bypass LLM pour listes) ───────────────

    def _structured_query(self, intent_data: dict) -> Optional[List[Dict]]:
        """Route une question vers DuckDB si la source est tabulaire.
        Retourne None si la source est un document non-structuré (.docx/.pdf)
        ou si elle n'existe pas → le pipeline RAG classique prend le relais.
        """
        source = intent_data.get("source")
        column = intent_data.get("column")
        filt = intent_data.get("filter") or {}

        if not source or source not in self.schema:
            return None
        if self.schema[source].get("is_doc"):
            return None  # documents texte → pipeline LLM classique
        if not self.structured.has_table(source):
            return None  # source dans le schéma mais pas dans DuckDB (cohérence)

        results = self.structured.list_values(
            table=source, column=column, filters=filt, distinct=True,
        )
        return results if results else None

    @staticmethod
    def _fold(text: str) -> str:
        """Normalise pour comparaison : minuscules + suppression d'accents."""
        text = text.lower()
        for src, dst in [("é","e"),("è","e"),("ê","e"),("ë","e"),
                         ("à","a"),("â","a"),("ä","a"),
                         ("î","i"),("ï","i"),("ô","o"),("ö","o"),
                         ("ù","u"),("û","u"),("ü","u"),("ç","c"),
                         ("œ","oe"),("æ","ae")]:
            text = text.replace(src, dst)
        return text

    # ── VALIDATION LLM PAR BATCHES (filtrage des résultats d'extraction) ────

    def _llm_validate_batch(
        self,
        question: str,
        items: List[str],
        batch_size: int = 10,
    ) -> List[str]:
        """Filtre une liste de candidats avec le LLM, batch par batch.

        Pour chaque batch de N éléments numérotés, demande au LLM quels
        numéros répondent réellement à la question. L'ordre est préservé.
        En cas d'échec d'un batch (parse, timeout) → on conserve le batch
        intact (failsafe : ne jamais perdre silencieusement de données).
        """
        if not items:
            return items

        kept: List[str] = []
        n_batches = (len(items) + batch_size - 1) // batch_size
        logger.info("  ⟹ Validation LLM: %d éléments en %d batch(es) de %d max",
                    len(items), n_batches, batch_size)

        for b_idx in range(n_batches):
            start = b_idx * batch_size
            batch = items[start:start + batch_size]
            prompt = self._build_validation_prompt(question, batch)
            try:
                raw = self.llm.generate(
                    prompt=prompt,
                    system=("Tu es un filtre strict. Retourne UNIQUEMENT les numéros "
                            "des éléments qui répondent EXACTEMENT à la question, "
                            "séparés par des virgules. 0 si aucun."),
                    temperature=0.0,
                    max_tokens=60,
                )
                indices = self._parse_kept_indices(raw, len(batch))
                if indices is None:  # parse échoué
                    logger.warning("    Batch %d/%d: parse échoué (%r) — conservé",
                                   b_idx + 1, n_batches, raw[:80])
                    kept.extend(batch)
                else:
                    selected = [batch[i] for i in indices]
                    logger.info("    Batch %d/%d: %d/%d retenus",
                                b_idx + 1, n_batches, len(selected), len(batch))
                    kept.extend(selected)
            except Exception as e:
                logger.warning("    Batch %d/%d: exception (%s) — conservé",
                               b_idx + 1, n_batches, e)
                kept.extend(batch)

        logger.info("  ⟹ Validation LLM terminée: %d/%d éléments retenus",
                    len(kept), len(items))
        return kept

    @staticmethod
    def _build_validation_prompt(question: str, batch: List[str]) -> str:
        numbered = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(batch))
        return (
            f"Question utilisateur:\n{question}\n\n"
            f"Candidats à valider:\n{numbered}\n\n"
            f"Tâche: identifie les numéros des candidats qui répondent EXACTEMENT "
            f"à la question (rejette les hors-sujet, les doublons orthographiques, "
            f"les fautes de frappe). Retourne UNIQUEMENT les numéros séparés par "
            f"des virgules. Si aucun ne convient, retourne 0.\n\n"
            f"Numéros retenus:"
        )

    @staticmethod
    def _parse_kept_indices(raw: str, batch_size: int) -> Optional[List[int]]:
        """Parse la réponse LLM en liste d'indices 0-based.
        Retourne None si la réponse est inintelligible (aucun nombre trouvé).
        Retourne [] si le LLM répond '0' (= aucun retenu)."""
        if not raw or not raw.strip():
            return None
        nums = re.findall(r"\d+", raw)
        if not nums:
            return None
        # Si la réponse contient uniquement '0' → aucun retenu
        unique_nums = {int(n) for n in nums}
        if unique_nums == {0}:
            return []
        indices: List[int] = []
        seen: set = set()
        for n in nums:
            i = int(n) - 1  # 1-indexed → 0-indexed
            if 0 <= i < batch_size and i not in seen:
                seen.add(i)
                indices.append(i)
        return indices

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

        # 1. Classification d'intent (1 appel LLM léger, mis en cache)
        intent_data = self.intent_router.classify(question)
        exhaustive = intent_data["exhaustive"]
        source = intent_data["source"]
        column = intent_data["column"]

        # 2. Bypass LLM : requête SQL directe pour les sources tabulaires
        if exhaustive and source:
            direct = self._structured_query(intent_data)
            if direct:
                # Dédup case-insensitive + accent-insensitive (fusionne SOCIALE/SOCIALES/SOCIOAL)
                seen: set = set()
                unique_items = []
                for d in direct:
                    item = d["content"].rstrip(".").strip()
                    key = self._fold(item)
                    if key not in seen and len(item) > 2:
                        seen.add(key)
                        unique_items.append(item)

                # Validation LLM par batches : filtre hors-sujet et doublons sémantiques
                validated = self._llm_validate_batch(
                    question=question,
                    items=unique_items,
                    batch_size=self.config.validation_batch_size,
                ) if getattr(self.config, "validation_enabled", True) else unique_items

                answer = (f"Il y a {len(validated)} résultats :\n"
                          + "\n".join(f"{i+1}. {item}"
                                      for i, item in enumerate(validated)))
                elapsed = round(time.time() - start, 2)
                sources = list({d["metadata"].get("filename", "?") for d in direct})
                logger.info("  ✅ Réponse directe (avec validation LLM): %d éléments en %.2fs",
                            len(validated), elapsed)
                return {
                    "question":        question,
                    "search_query":    question,
                    "answer":          answer,
                    "sources":         sources,
                    "chunks_used":     len(validated),
                    "elapsed_seconds": elapsed,
                    "intent":          intent_data,
                }

        # 3. Transformation de la requête (optionnelle)
        if use_query_transform:
            logger.info("  [1/4] Transformation de la requête...")
            search_query = self.query_transformer.rewrite(question)
            if search_query != question:
                logger.info("        → %s", search_query)
        else:
            search_query = question

        # 4. Recherche hybride + Reranking (avec cache)
        cache_key = (search_query.strip().lower(), exhaustive, source, column)
        if cache_key in self._retrieval_cache:
            logger.info("  [2/4] Recherche hybride... (cache hit)")
            logger.info("  [3/4] Reranking... (cache hit)")
            reranked = self._retrieval_cache[cache_key]
        else:
            logger.info("  [2/4] Recherche hybride%s...", " (mode exhaustif)" if exhaustive else "")
            query_emb = self.embedder.embed_single(search_query)
            if exhaustive:
                k_dense = min(self.config.max_chunks_exhaustive * 5, self.vector_store.count())
                k_sparse = self.config.max_chunks_exhaustive * 5
            else:
                k_dense = self.config.top_k_dense
                k_sparse = self.config.top_k_sparse
            dense  = self.vector_store.search(query_emb, k=k_dense)
            sparse = self.bm25.search(search_query, k=k_sparse)
            hybrid = reciprocal_rank_fusion(dense, sparse, k=self.config.rrf_k)
            logger.info("        Dense: %d | Sparse: %d | RRF: %d", len(dense), len(sparse), len(hybrid))

            logger.info("  [3/4] Reranking...")
            if exhaustive:
                reranked = self.reranker.rerank(
                    query=search_query,
                    documents=hybrid[:self.config.max_chunks_exhaustive * 3],
                    top_k=self.config.max_chunks_exhaustive,
                )
                if reranked:
                    logger.info("        Scores reranker: [%.2f ... %.2f]",
                                reranked[0].get("rerank_score", 0),
                                reranked[-1].get("rerank_score", 0))
            else:
                reranked = self.reranker.rerank(
                    query=search_query,
                    documents=hybrid[:20],
                    top_k=self.config.top_k_after_rerank,
                )
            if len(self._retrieval_cache) >= self._cache_max_size:
                oldest_key = next(iter(self._retrieval_cache))
                del self._retrieval_cache[oldest_key]
            self._retrieval_cache[cache_key] = reranked
        logger.info("        → %d chunks retenus", len(reranked))

        # 5. Filtre par source pertinente
        if source:
            reranked = self._filter_by_source(reranked, source)

        # 6. Génération
        logger.info("  [4/4] Génération LLM...")
        context      = self._format_context(reranked)
        history_text = self._format_history(history) if history else ""
        template = _GENERATION_PROMPT_LIST if exhaustive else _GENERATION_PROMPT
        prompt = template.format(
            context=context,
            question=question,
            history=history_text,
        )

        max_tokens = self.config.llm_max_tokens_long if exhaustive else self.config.llm_max_tokens

        if stream:
            answer_parts = []
            for token in self.llm.generate_stream(
                prompt=prompt, system=_SYSTEM_PROMPT, max_tokens=max_tokens,
            ):
                print(token, end="", flush=True)
                answer_parts.append(token)
            print()
            answer = "".join(answer_parts)
        else:
            answer = self.llm.generate(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                temperature=self.config.llm_temperature,
                max_tokens=max_tokens,
            )

        elapsed = round(time.time() - start, 2)

        return {
            "question":        question,
            "search_query":    search_query,
            "answer":          answer,
            "sources":         self._extract_sources(reranked),
            "chunks_used":     len(reranked),
            "elapsed_seconds": elapsed,
            "intent":          intent_data,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = []
        for msg in history[-4:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "Historique:\n" + "\n".join(lines) + "\n\n"

    def _format_context(self, chunks: List[Dict]) -> str:
        """Formate le contexte de façon compacte pour les petits modèles."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(chunk['content'])
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
