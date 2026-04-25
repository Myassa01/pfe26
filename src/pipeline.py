"""Pipeline RAG principal — orchestration de toutes les couches."""
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

# ── Prompts ──────────────────────────────────────────────────────────────────

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

# ── Mots-clés liste exhaustive ────────────────────────────────────────────────

_LIST_KEYWORDS = [
    "donne moi les", "donne-moi les", "donnes moi les",
    "donne les", "donne la liste",
    "quels sont", "quelles sont",
    "qui sont les",
    "affiche les", "montre les", "cite les",
    "liste", "lister", "tous les", "toutes les", "tout le", "toute la",
    "combien", "enumere", "enumerer", "ensemble des", "totalite",
    "chaque", "l'ensemble", "recapitulatif", "recapitule",
    "affiche tous", "affiche toutes", "montre tous", "montre toutes",
    "disponible", "disponibles", "existant", "existants",
    # formations
    "formation", "formations", "plan de formation",
    "formations disponibles", "formations disponible",
    "formations obligatoires", "formation obligatoire",
    "formations facultatives", "formation facultative",
    "obligatoire", "obligatoires", "facultatif", "facultatifs", "facultatives",
]


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
        self.llm = HFClient(model=config.llm_model)
        self.query_transformer = QueryTransformer(llm=self.llm)

        self._retrieval_cache: Dict[tuple, List[Dict]] = {}
        self._cache_max_size: int = 128

        logger.info("=" * 50)
        logger.info("Pipeline prêt.")

    # ── Normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_q(q: str) -> str:
        q = q.lower()
        for src, dst in [("é","e"),("è","e"),("ê","e"),("ë","e"),
                         ("à","a"),("â","a"),("ä","a"),("î","i"),
                         ("ï","i"),("ô","o"),("ù","u"),("û","u"),("ü","u")]:
            q = q.replace(src, dst)
        return q

    @staticmethod
    def _is_list_question(question: str) -> bool:
        q = RAGPipeline._normalize_q(question)
        normalized_keywords = [RAGPipeline._normalize_q(kw) for kw in _LIST_KEYWORDS]
        return any(kw in q for kw in normalized_keywords)

    # ── Source keyword mapping ────────────────────────────────────────────────

    _SOURCE_KEYWORDS = {
        "DIRECTION":          ["directeur", "directeurs", "directrice", "directrices", "direction", "directions"],
        "DEPARTEMENT":        ["departement", "departements", "département", "départements",
                               "chef de departement", "chefs de departement"],
        "SERVICE":            ["service", "services", "chef de service", "chefs de service"],
        "KAM_FORMATIONS_GTP": ["formation", "formations", "plan de formation",
                               "obligatoire", "obligatoires", "facultatif", "facultatifs", "facultatives"],
    }

    _EXCLUDE_POSTE_KEYWORDS = [
        "chantier", "chantiers", "affectation", "affectations",
        "matricule", "matricules", "nom", "prenom", "prénom",
        "observation", "fonction",
    ]

    @staticmethod
    def _detect_relevant_sources(question: str) -> set:
        q = question.lower()
        relevant = set()
        for source, keywords in RAGPipeline._SOURCE_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                relevant.add(source)
        return relevant

    @staticmethod
    def _should_exclude_poste(question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in RAGPipeline._EXCLUDE_POSTE_KEYWORDS)

    @staticmethod
    def _filter_by_source(chunks: list, relevant_sources: set, exclude_poste: bool = False) -> list:
        filtered = chunks

        if exclude_poste and not relevant_sources:
            filtered = [c for c in filtered
                        if c["metadata"].get("filename", "").upper().rsplit(".", 1)[0] != "POSTE"]
            if filtered:
                logger.info("  → Exclusion POSTE: %d/%d chunks retenus", len(filtered), len(chunks))
                return filtered
            return chunks

        if not relevant_sources:
            return chunks

        from_relevant, from_other = [], []
        for chunk in filtered:
            fname = chunk["metadata"].get("filename", "").upper()
            stem  = fname.rsplit(".", 1)[0] if "." in fname else fname
            if stem in relevant_sources:
                from_relevant.append(chunk)
            else:
                from_other.append(chunk)

        if not from_relevant:
            return chunks

        result = from_relevant + from_other[:3]
        logger.info("  → Filtre source: %d/%d chunks retenus (sources: %s)",
                    len(result), len(chunks), ", ".join(relevant_sources))
        return result

    # ── Détection chunk formations ────────────────────────────────────────────

    _FORMATIONS_STEMS = {
        "KAM_FORMATIONS_GTP", "FORMATIONS_GTP",
        "KAM_FORMATION_GTP",  "FORMATIONS", "FORMATION_GTP",
    }

    def _is_formations_chunk(self, content: str, fname: str) -> bool:
        """Double vérification : nom de fichier OU marqueur dans le contenu."""
        stem = fname.upper().rsplit(".", 1)[0] if "." in fname else fname.upper()
        name_ok    = stem in self._FORMATIONS_STEMS or "FORMATION" in stem
        content_ok = ("[OBLIGATOIRE]" in content or "[FACULTATIVE]" in content
                      or "FORMATIONS OBLIGATOIRES" in content.upper()
                      or "FORMATIONS FACULTATIVES" in content.upper())
        return name_ok or content_ok

    # ── CORRECTIF 2 : Nettoyage du format de sortie des formations ────────────

    @staticmethod
    def _clean_formation_item(raw: str) -> str:
        """
        Transforme le format brut d'un chunk formation en affichage propre.

        Entrée  : "[OBLIGATOIRE] N°01 — Gestion des Déchets | Statut: Obligatoire"
        Sortie  : "Gestion des Déchets (Obligatoire)"

        Entrée  : "[FACULTATIVE] N°41 — Rapport de Gestion | Statut: Facultative"
        Sortie  : "Rapport de Gestion (Facultative)"
        """
        s = raw.strip()

        # Déterminer le statut depuis le préfixe
        if s.startswith("[OBLIGATOIRE]"):
            statut = "Obligatoire"
        elif s.startswith("[FACULTATIVE]"):
            statut = "Facultative"
        else:
            # Pas un chunk formation reconnu → retourner tel quel
            return s

        # Retirer le préfixe [OBLIGATOIRE] / [FACULTATIVE]
        s = re.sub(r"^\[(OBLIGATOIRE|FACULTATIVE)\]\s*", "", s)

        # Retirer "N°XX — " ou "N°XX- " en début
        s = re.sub(r"^N°\d+\s*[—\-–]\s*", "", s)

        # Retirer la partie " | Statut: ..." en fin
        s = re.sub(r"\s*\|\s*Statut\s*:.*$", "", s, flags=re.IGNORECASE)

        titre = s.strip()

        return f"{titre} ({statut})"

    # ── Extraction directe formations (bypass LLM complet) ───────────────────

    def _try_direct_extract_formations(self, question: str) -> Optional[List[Dict]]:
        """
        Bypass complet du RAG pour les questions sur les formations.

        Les marqueurs [OBLIGATOIRE] et [FACULTATIVE] sont produits par
        loader._load_excel_formations() — UN chunk par formation.

        Retourne None si la question ne porte pas sur les formations,
        laissant le flux normal prendre le relais.
        """
        q = self._normalize_q(question)

        # ── CORRECTIF 1a : ask_all — ne pas inclure les variantes avec type précis ──
        ask_all = any(kw in q for kw in [
            "formations disponibles", "formations disponible",
            "quelles sont les formations", "quels sont les formations",
            "liste des formations", "toutes les formations",
            "plan de formation",
            "formations existantes",
            "formations de l'entreprise", "formations dans cette entreprise",
            "formations disponibles dans", "quelles formations",
            "formation qui existe",
            # NOTE : "formations qui existe" retiré car ambigu avec "obligatoires qui existent"
        ])

        # ── CORRECTIF 1b : ask_obligatoire — couvre toutes les variantes avec/sans s ──
        ask_obligatoire = any(kw in q for kw in [
            "formations obligatoires",   # pluriel complet
            "formations obligatoire",    # pluriel + singulier statut  ← NOUVEAU
            "formation obligatoire",     # singulier
            "formation obligatoires",    # singulier + pluriel statut  ← NOUVEAU
            "obligatoires",              # adjectif seul pluriel
            "obligatoire",               # adjectif seul singulier     ← NOUVEAU
        ])

        # ── CORRECTIF 1b : ask_facultatif — idem ──
        ask_facultatif = any(kw in q for kw in [
            "formations facultatives",
            "formations facultative",    # ← NOUVEAU
            "formation facultative",
            "formation facultatives",    # ← NOUVEAU
            "facultatif", "facultatifs",
            "facultatives", "facultative",
        ])

        # ── CORRECTIF 1c : la spécificité prime TOUJOURS sur ask_all ─────────
        # Si l'utilisateur précise un type, ask_all est ignoré même s'il a matché
        if ask_obligatoire or ask_facultatif:
            ask_all = False

        if not (ask_all or ask_obligatoire or ask_facultatif):
            return None

        # Éviter l'interférence avec d'autres questions RH
        q_lower = question.lower()
        if any(kw in q_lower for kw in ["directeur", "chef de", "departement",
                                         "département", "chantier", "matricule",
                                         "salaire", "badge"]):
            return None

        results = []
        for doc in self.bm25.documents:
            fname   = doc.metadata.get("filename", "")
            content = doc.content

            if not self._is_formations_chunk(content, fname):
                continue

            # Ignorer les chunks d'en-tête (pas des formations individuelles)
            chunk_type = doc.metadata.get("chunk_type", "formation")
            if chunk_type in ("global_header", "section_header"):
                continue

            content_upper = content.upper()

            # ── Filtre par type ────────────────────────────────────────
            if ask_obligatoire and not ask_all:
                if "[OBLIGATOIRE]" not in content_upper:
                    continue
            elif ask_facultatif and not ask_all:
                if "[FACULTATIVE]" not in content_upper:
                    continue
            # ask_all → pas de filtre

            results.append({"content": content, "metadata": doc.metadata})

        if results:
            logger.info(
                "  ⟹ Extraction directe formations: %d chunks "
                "(all=%s, obligatoire=%s, facultatif=%s)",
                len(results), ask_all, ask_obligatoire, ask_facultatif,
            )
            return results

        logger.warning(
            "  ⚠ Formations demandées mais aucun chunk trouvé.\n"
            "    → Vérifier l'ingestion de KAM_Formations_GTP.xlsx\n"
            "    → Vérifier que loader.py produit bien [OBLIGATOIRE]/[FACULTATIVE]"
        )
        return None

    # ── Extraction directe générique ──────────────────────────────────────────

    _DIRECT_EXTRACT_PATTERNS = [
        (["chef de departement", "chefs de departement",
          "chef de département", "chefs de département"],           None,       ["DEPARTEMENT"]),
        (["chef de service", "chefs de service"],                   None,       ["SERVICE"]),
        (["directeur", "directeurs", "directrice"],                 None,       ["DIRECTION"]),
        (["chantier", "chantiers"],                                 "CHANTIER", ["DIRECTION", "DEPARTEMENT", "SERVICE"]),
        (["service"],                                               "CHANTIER", ["SERVICE"]),
        (["departement", "département"],                            "CHANTIER", ["DEPARTEMENT"]),
        (["direction"],                                             "CHANTIER", ["DIRECTION"]),
    ]

    def _try_direct_extract(self, question: str) -> Optional[List[Dict]]:
        q = self._normalize_q(question)

        # 1. Bypass formations en priorité
        formations = self._try_direct_extract_formations(question)
        if formations is not None:
            return formations

        # 2. Patterns génériques
        for keywords, column, sources in self._DIRECT_EXTRACT_PATTERNS:
            if not any(kw in q for kw in keywords):
                continue

            results    = []
            seen_vals: set = set()

            for doc in self.bm25.documents:
                fname = doc.metadata.get("filename", "").upper()
                stem  = fname.rsplit(".", 1)[0] if "." in fname else fname
                if stem not in sources:
                    continue

                if column:
                    for part in doc.content.split("|"):
                        part = part.strip()
                        if ":" in part:
                            key, val = part.split(":", 1)
                            key = key.strip().lstrip("[").rstrip("]").strip()
                            val = val.strip()
                            if key.upper() == column and val and val.lower() not in seen_vals:
                                seen_vals.add(val.lower())
                                results.append({"content": val, "metadata": doc.metadata})
                else:
                    results.append({"content": doc.content, "metadata": doc.metadata})

            if results:
                logger.info("  ⟹ Extraction directe: %d résultats (colonne=%s, sources=%s)",
                            len(results), column or "ALL", ",".join(sources))
                return results

        return None

    # ── Déduplication ─────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_chunks(chunks: list) -> list:
        seen: set = set()
        unique = []
        for chunk in chunks:
            h = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(chunk)
        removed = len(chunks) - len(unique)
        if removed:
            logger.info("  → %d chunk(s) dupliqué(s) supprimé(s)", removed)
        return unique

    # ── Ingestion ─────────────────────────────────────────────────────────────

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

        logger.info("[2/4] Découpage (chunk_size=%d, overlap=%d)...",
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
        texts      = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts, batch_size=self.config.embedding_batch_size, show_progress=True)
        logger.info("  → Shape: %s", embeddings.shape)

        logger.info("[4/4] Indexation (ChromaDB + BM25)...")
        self.vector_store.add(
            ids=[c.id for c in chunks],
            embeddings=list(embeddings),
            texts=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        bm25_docs = [BM25Document(id=c.id, content=c.content, metadata=c.metadata) for c in chunks]
        self.bm25.add_documents(bm25_docs)
        self.bm25.save(self.config.bm25_index_path)
        self._retrieval_cache.clear()

        logger.info("=" * 50)
        logger.info("Ingestion terminée: %d chunks indexés", len(chunks))
        logger.info("=" * 50)
        return {"documents": len(documents), "chunks": len(chunks), "embedding_dim": int(embeddings.shape[1])}

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            embedding_model=self.config.embedding_model,
        )
        if not chunks:
            return {"documents": len(documents), "chunks": 0, "embedding_dim": 0}
        chunks     = self._deduplicate_chunks(chunks)
        texts      = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts, batch_size=self.config.embedding_batch_size, show_progress=False)
        self.vector_store.add(
            ids=[c.id for c in chunks],
            embeddings=list(embeddings),
            texts=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        bm25_docs = [BM25Document(id=c.id, content=c.content, metadata=c.metadata) for c in chunks]
        self.bm25.add_documents(bm25_docs)
        self.bm25.save(self.config.bm25_index_path)
        self._retrieval_cache.clear()
        return {"documents": len(documents), "chunks": len(chunks), "embedding_dim": int(embeddings.shape[1])}

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        use_query_transform: bool = False,
        stream: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        start      = time.time()
        exhaustive = self._is_list_question(question)
        if exhaustive:
            logger.info("  ⟹ Mode exhaustif détecté")

        # ── Bypass direct ─────────────────────────────────────────────────
        if exhaustive:
            direct = self._try_direct_extract(question)
            if direct:
                # ── CORRECTIF 2 : nettoyage du format d'affichage ──────────
                # Détecter si les résultats sont des formations (contiennent [OBLIGATOIRE]/[FACULTATIVE])
                is_formations = any(
                    d["content"].startswith("[OBLIGATOIRE]") or d["content"].startswith("[FACULTATIVE]")
                    for d in direct
                )

                seen: set = set()
                unique_items = []
                for d in direct:
                    raw = d["content"].rstrip(".").strip()

                    # Nettoyer uniquement les formations, pas les autres types
                    if is_formations:
                        item = self._clean_formation_item(raw)
                    else:
                        item = raw

                    key = item.lower()
                    if key not in seen and len(item) > 2:
                        seen.add(key)
                        unique_items.append(item)

                answer  = (f"Il y a {len(unique_items)} résultats :\n"
                           + "\n".join(f"{i+1}. {item}" for i, item in enumerate(unique_items)))
                elapsed = round(time.time() - start, 2)
                sources = list({d["metadata"].get("filename", "?") for d in direct})
                logger.info("  ✅ Réponse directe (sans LLM): %d éléments en %.2fs", len(unique_items), elapsed)
                return {
                    "question":        question,
                    "search_query":    question,
                    "answer":          answer,
                    "sources":         sources,
                    "chunks_used":     len(unique_items),
                    "elapsed_seconds": elapsed,
                }

        # ── Transformation de requête ──────────────────────────────────────
        if use_query_transform:
            logger.info("  [1/4] Transformation de la requête...")
            search_query = self.query_transformer.rewrite(question)
            if search_query != question:
                logger.info("        → %s", search_query)
        else:
            search_query = question

        # ── Recherche hybride + Reranking (avec cache) ────────────────────
        cache_key = (search_query.strip().lower(), exhaustive)
        if cache_key in self._retrieval_cache:
            logger.info("  [2+3/4] Cache hit")
            reranked = self._retrieval_cache[cache_key]
        else:
            logger.info("  [2/4] Recherche hybride%s...", " (mode exhaustif)" if exhaustive else "")
            query_emb = self.embedder.embed_single(search_query)

            if exhaustive:
                k_dense  = min(self.config.max_chunks_exhaustive * 5, self.vector_store.count())
                k_sparse = self.config.max_chunks_exhaustive * 5
            else:
                k_dense  = self.config.top_k_dense
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
            else:
                reranked = self.reranker.rerank(
                    query=search_query,
                    documents=hybrid[:20],
                    top_k=self.config.top_k_after_rerank,
                )

            if len(self._retrieval_cache) >= self._cache_max_size:
                del self._retrieval_cache[next(iter(self._retrieval_cache))]
            self._retrieval_cache[cache_key] = reranked

        logger.info("        → %d chunks retenus", len(reranked))

        # ── Filtre par source ──────────────────────────────────────────────
        relevant_sources = self._detect_relevant_sources(question)
        exclude_poste    = self._should_exclude_poste(question)
        if relevant_sources or exclude_poste:
            reranked = self._filter_by_source(reranked, relevant_sources, exclude_poste)

        # ── Génération LLM ────────────────────────────────────────────────
        logger.info("  [4/4] Génération LLM...")
        context      = self._format_context(reranked)
        history_text = self._format_history(history) if history else ""
        template     = _GENERATION_PROMPT_LIST if exhaustive else _GENERATION_PROMPT
        prompt       = template.format(context=context, question=question, history=history_text)
        max_tokens   = self.config.llm_max_tokens_long if exhaustive else self.config.llm_max_tokens

        if stream:
            answer_parts = []
            for token in self.llm.generate_stream(prompt=prompt, system=_SYSTEM_PROMPT, max_tokens=max_tokens):
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
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = []
        for msg in history[-4:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "Historique:\n" + "\n".join(lines) + "\n\n"

    def _format_context(self, chunks: List[Dict]) -> str:
        return "\n\n---\n\n".join(c["content"] for c in chunks)

    def _extract_sources(self, chunks: List[Dict]) -> List[str]:
        seen: set = set()
        sources = []
        for chunk in chunks:
            src = chunk["metadata"].get("filename", "inconnu")
            if src not in seen:
                sources.append(src)
                seen.add(src)
        return sources
