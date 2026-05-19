"""
Nœuds LangGraph pour le pipeline RAG Myassa.

Chaque nœud : (state: GraphState) -> dict  (mise à jour partielle de l'état)

Les prompts et helpers autrefois dans pipeline.py sont ici au niveau module.
"""
import logging
import re
from functools import partial
from typing import Any, Dict, List, Optional

from .state import GraphState
from ..retrieval.hybrid_search import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Tu es un assistant RH. Réponds UNIQUEMENT en français, de façon COURTE et DIRECTE.
Règles :
1. Utilise UNIQUEMENT le contexte fourni. Si l'information est absente → "Information non disponible."
2. N'invente rien. Une seule phrase pour les questions sur une personne ou un poste."""

_GENERATION_PROMPT = """<contexte>
{context}
</contexte>

{history}
<question>
{question}
</question>

INSTRUCTIONS :
- Si "qui est [NOM]" → "[NOM COMPLET] est [FONCTION]."
- Si "qui est le [POSTE]" → "Le [POSTE] est [NOM COMPLET]."
- Une seule phrase. Si non trouvé → "Information non disponible."

Réponse :"""

_GENERATION_PROMPT_LIST = """<contexte>
{context}
</contexte>

{history}
<question>
{question}
</question>

INSTRUCTIONS :
- Liste numérotée, un élément par ligne, uniquement les noms présents dans le contexte.
- Pas d'explication ni de commentaire.

Réponse :"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS MODULE-LEVEL
# ─────────────────────────────────────────────────────────────────────────────

def _format_history(history: Optional[List[Dict]]) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "Historique de la conversation:\n" + "\n".join(lines) + "\n\n"


def _format_context(chunks: List[Dict]) -> str:
    return "\n\n---\n\n".join(c["content"] for c in chunks)


def _extract_sources(chunks: List[Dict]) -> List[str]:
    seen: set = set()
    sources = []
    for chunk in chunks:
        src = chunk["metadata"].get("filename", "inconnu")
        if src not in seen:
            sources.append(src)
            seen.add(src)
    return sources


def _fold(text: str) -> str:
    text = text.lower()
    for src, dst in [
        ("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
        ("à", "a"), ("â", "a"), ("ä", "a"),
        ("î", "i"), ("ï", "i"), ("ô", "o"), ("ö", "o"),
        ("ù", "u"), ("û", "u"), ("ü", "u"), ("ç", "c"),
        ("œ", "oe"), ("æ", "ae"),
    ]:
        text = text.replace(src, dst)
    return text


def _normalize_stem(fname: str) -> str:
    stem = fname.rsplit(".", 1)[0] if "." in fname else fname
    stem = stem.upper().strip()
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    stem = re.sub(r"\s*_\d+\s*$", "", stem)
    return stem.strip()


def _filter_by_source(chunks: List[Dict], source: Optional[str]) -> List[Dict]:
    if not source:
        return chunks
    from_relevant, from_other = [], []
    for chunk in chunks:
        fname = chunk["metadata"].get("filename", "")
        if _normalize_stem(fname) == source:
            from_relevant.append(chunk)
        else:
            from_other.append(chunk)
    if not from_relevant:
        return chunks
    result = from_relevant + from_other[:3]
    logger.info("  → Filtre source: %d/%d chunks retenus (source: %s)",
                len(result), len(chunks), source)
    return result


def _extract_primary_value(item: str, source: Optional[str], structured_engine) -> str:
    content = item
    if "] " in content and content.startswith("["):
        content = content.split("] ", 1)[1]

    pairs: Dict[str, str] = {}
    for part in content.split(" | "):
        part = part.strip()
        if ": " in part:
            k, v = part.split(": ", 1)
            pairs[k.strip().upper()] = v.strip().rstrip(".")
        elif part:
            pairs.setdefault("__RAW__", part.strip())

    if not pairs:
        return item

    if source and structured_engine.has_table(source):
        primary_col = structured_engine.get_primary_column(source)
        if primary_col:
            val = pairs.get(primary_col.upper())
            if val and len(val) > 2:
                return val

    best = max(pairs.values(), key=lambda v: len(v), default=item)
    return best if len(best) > 2 else item


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CONDITIONNELLE
# ─────────────────────────────────────────────────────────────────────────────

def route_node(state: GraphState, schema: Dict[str, dict], structured) -> str:
    """Retourne la clé de branche selon l'intent classifié."""
    intent_data = state.get("intent_data", {})
    exhaustive  = intent_data.get("exhaustive", False)
    source      = intent_data.get("source")
    filt        = intent_data.get("filter") or {}

    # Chemin A : liste exhaustive → SQL direct
    if exhaustive and source:
        if source in schema and not schema[source].get("is_doc"):
            if structured.has_table(source):
                return "exhaustive_path"

    # Chemin B : QA structurée → SQL + LLM
    if (
        not exhaustive
        and source
        and not schema.get(source, {}).get("is_doc")
        and filt
        and structured.has_table(source)
    ):
        return "structured_qa_path"

    # Chemin C : RAG sémantique
    return "rag_path"


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY DE NŒUDS
# ─────────────────────────────────────────────────────────────────────────────

def build_nodes(components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crée tous les nœuds par closure sur les composants.

    Clés attendues dans components :
        query_transformer, intent_router, embedder, vector_store,
        bm25, reranker, llm, structured, schema, config
    """
    query_transformer = components["query_transformer"]
    intent_router     = components["intent_router"]
    embedder          = components["embedder"]
    vector_store      = components["vector_store"]
    bm25              = components["bm25"]
    reranker          = components["reranker"]
    llm               = components["llm"]
    structured        = components["structured"]
    config            = components["config"]

    # ── 1. contextualize_node ─────────────────────────────────────────────

    def contextualize_node(state: GraphState) -> dict:
        question = state["question"]
        history  = state.get("history")
        resolved = query_transformer.contextualize(question, history)
        if resolved != question:
            logger.info("  [ctx] Question reformulée: %r → %r", question, resolved)
        return {"resolved_question": resolved}

    # ── 2. intent_node ────────────────────────────────────────────────────

    def intent_node(state: GraphState) -> dict:
        intent_data = intent_router.classify(state["resolved_question"])
        return {"intent_data": intent_data}

    # ── 3. exhaustive_node ────────────────────────────────────────────────

    def exhaustive_node(state: GraphState) -> dict:
        intent_data = state["intent_data"]
        source      = intent_data.get("source")
        column      = intent_data.get("column")
        filt        = intent_data.get("filter") or {}

        direct = structured.list_values(
            table=source, column=column, filters=filt, distinct=True,
        )
        sql_warnings = list(structured.last_warnings)

        if not direct:
            return {
                "answer":      "Aucun résultat trouvé.",
                "sources":     [],
                "chunks_used": 0,
                "warnings":    sql_warnings,
                "path_taken":  "exhaustive",
            }

        seen: set = set()
        unique_names = []
        for d in direct:
            raw  = d["content"].rstrip(".").strip()
            name = _extract_primary_value(raw, source, structured)
            key  = _fold(name)
            if key not in seen and len(name) > 2:
                seen.add(key)
                unique_names.append(name)

        prefix_lines = []
        if sql_warnings:
            prefix_lines.append("⚠ Note :")
            for w in sql_warnings:
                prefix_lines.append(f"  • {w}")
            prefix_lines.append("")

        body = (
            f"Il y a {len(unique_names)} résultat(s) :\n"
            + "\n".join(f"{i+1}. {name}" for i, name in enumerate(unique_names))
        )
        answer  = "\n".join(prefix_lines) + body if prefix_lines else body
        sources = list({d["metadata"].get("filename", "?") for d in direct})

        logger.info("  ✅ Exhaustive: %d éléments", len(unique_names))
        return {
            "answer":      answer,
            "sources":     sources,
            "chunks_used": len(unique_names),
            "warnings":    sql_warnings,
            "path_taken":  "exhaustive",
        }

    # ── 4. structured_qa_node ────────────────────────────────────────────

    def structured_qa_node(state: GraphState) -> dict:
        intent_data       = state["intent_data"]
        resolved_question = state["resolved_question"]
        history           = state.get("history")
        source            = intent_data.get("source")
        column            = intent_data.get("column")
        filt              = intent_data.get("filter") or {}

        qa_rows      = structured.list_values(table=source, column=column, filters=filt, distinct=True)
        sql_warnings = list(structured.last_warnings)

        context      = "\n".join(r["content"] for r in qa_rows[:20])
        history_text = _format_history(history)
        prompt = _GENERATION_PROMPT.format(
            context=context, question=resolved_question, history=history_text,
        )
        answer = llm.generate(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        sources = list({r["metadata"].get("filename", "?") for r in qa_rows})
        logger.info("  ✅ Structured QA: %d ligne(s) SQL → LLM", len(qa_rows))
        return {
            "answer":      answer,
            "sources":     sources,
            "chunks_used": len(qa_rows),
            "warnings":    sql_warnings,
            "path_taken":  "structured_qa",
        }

    # ── 5. retrieve_node ─────────────────────────────────────────────────

    def retrieve_node(state: GraphState) -> dict:
        intent_data  = state["intent_data"]
        exhaustive   = intent_data.get("exhaustive", False)
        search_query = state["resolved_question"]

        if vector_store.count() == 0:
            logger.warning("  [retrieve] Vector store vide — aucun document indexé.")
            return {
                "search_query":   search_query,
                "hybrid_results": [],
                "answer":         "Aucun document n'est encore indexé. Veuillez lancer l'ingestion depuis l'onglet Documents.",
                "sources":        [],
                "chunks_used":    0,
                "path_taken":     "empty_index",
            }

        query_emb = embedder.embed_single(search_query)

        if exhaustive:
            k_dense  = min(config.max_chunks_exhaustive * 5, vector_store.count())
            k_sparse = config.max_chunks_exhaustive * 5
        else:
            k_dense  = config.top_k_dense
            k_sparse = config.top_k_sparse

        dense  = vector_store.search(query_emb, k=k_dense)
        sparse = bm25.search(search_query, k=k_sparse)
        hybrid = reciprocal_rank_fusion(
            dense, sparse,
            k=config.rrf_k,
            dense_weight=config.rrf_dense_weight,
            sparse_weight=config.rrf_sparse_weight,
        )

        logger.info("  [retrieve] Dense: %d | Sparse: %d | RRF: %d",
                    len(dense), len(sparse), len(hybrid))
        return {
            "search_query":   search_query,
            "hybrid_results": hybrid,
        }

    # ── 6. rerank_node ───────────────────────────────────────────────────

    def rerank_node(state: GraphState) -> dict:
        if state.get("path_taken") == "empty_index":
            return {"reranked_chunks": [], "filtered_chunks": []}

        intent_data  = state["intent_data"]
        exhaustive   = intent_data.get("exhaustive", False)
        intent       = intent_data.get("intent", "qa")
        source       = intent_data.get("source")
        search_query = state["search_query"]
        hybrid       = state["hybrid_results"]

        if exhaustive:
            reranked = reranker.rerank(
                query=search_query,
                documents=hybrid[:config.max_chunks_exhaustive * 3],
                top_k=config.max_chunks_exhaustive,
            )
        elif intent == "list":
            reranked = reranker.rerank(
                query=search_query,
                documents=hybrid[:20],
                top_k=min(config.top_k_after_rerank * 2, 10),
            )
        else:
            reranked = reranker.rerank(
                query=search_query,
                documents=hybrid[:20],
                top_k=config.top_k_after_rerank,
            )

        filtered = _filter_by_source(reranked, source)
        logger.info("  [rerank] → %d chunks retenus après filtre", len(filtered))
        return {
            "reranked_chunks": reranked,
            "filtered_chunks": filtered,
        }

    # ── 7. generate_node ─────────────────────────────────────────────────

    def generate_node(state: GraphState) -> dict:
        if state.get("path_taken") == "empty_index":
            return {"context": "", "answer": state.get("answer", ""), "path_taken": "empty_index"}

        intent_data       = state["intent_data"]
        exhaustive        = intent_data.get("exhaustive", False)
        resolved_question = state["resolved_question"]
        history           = state.get("history")
        filtered          = state["filtered_chunks"]

        context      = _format_context(filtered)
        history_text = _format_history(history)
        template     = _GENERATION_PROMPT_LIST if exhaustive else _GENERATION_PROMPT
        prompt       = template.format(
            context=context, question=resolved_question, history=history_text,
        )
        max_tokens = config.llm_max_tokens_long if exhaustive else config.llm_max_tokens

        answer = llm.generate(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=config.llm_temperature,
            max_tokens=max_tokens,
        )
        return {
            "context":    context,
            "answer":     answer,
            "path_taken": "semantic_rag",
        }

    # ── 8. finalize_node ─────────────────────────────────────────────────

    def finalize_node(state: GraphState) -> dict:
        path = state.get("path_taken")

        if path == "semantic_rag":
            filtered = state.get("filtered_chunks", [])
            sources  = _extract_sources(filtered)
            chunks   = len(filtered)
        elif path in ("exhaustive", "structured_qa"):
            sources = state.get("sources", [])
            chunks  = state.get("chunks_used", 0)
        elif path == "empty_index":
            # retrieve_node a court-circuité à cause d'un index vide
            sources = []
            chunks  = 0
        else:
            # Cas inattendu : tente les deux sources, garde ce qui est disponible
            filtered = state.get("filtered_chunks", [])
            if filtered:
                sources = _extract_sources(filtered)
                chunks  = len(filtered)
            else:
                sources = state.get("sources", [])
                chunks  = state.get("chunks_used", 0)
            logger.warning("finalize_node: path_taken inconnu (%r) — fallback heuristique", path)

        return {
            "sources":     sources,
            "chunks_used": chunks,
        }

    return {
        "contextualize_node":   contextualize_node,
        "intent_node":          intent_node,
        "exhaustive_node":      exhaustive_node,
        "structured_qa_node":   structured_qa_node,
        "retrieve_node":        retrieve_node,
        "rerank_node":          rerank_node,
        "generate_node":        generate_node,
        "finalize_node":        finalize_node,
    }
