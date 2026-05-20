"""
Nœuds LangGraph pour le pipeline RAG .

Chaque nœud : (state: GraphState) -> dict  (mise à jour partielle de l'état)

Les prompts et helpers autrefois dans pipeline.py sont ici au niveau module.
"""
import logging
import re
from functools import partial
from typing import Any, Dict, List, Optional

from .state import GraphState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Tu es un assistant RH francophone. Réponds en français de façon claire, concise et factuelle.
Règles :
1. Utilise UNIQUEMENT le contexte fourni. Si l'information est absente → "Information non disponible."
2. N'invente rien.
3. Ne réponds pas avec des procédures ou des détails génériques qui ne figurent pas explicitement dans le contexte.
4. Pour une question sur une personne ou un poste, réponds avec un énoncé court et précis.
5. Si tu ne peux pas vérifier la réponse dans le contexte, réponds "Information non disponible."."""

_GENERATION_PROMPT = """<contexte>
{context}
</contexte>

Sources du contexte : {sources}

{history}
<question>
{question}
</question>

INSTRUCTIONS :
- Réponds en français, en utilisant uniquement les éléments du contexte.
- Si l'historique de conversation est présent, utilise-le pour résoudre les références implicites ("son poste", "il", "ce département", etc.).
- Ne rédige pas de réponse générale basée sur des connaissances hors contexte.
- Si la question porte sur une personne ou un poste : "[NOM COMPLET] est [FONCTION]." ou "Le [POSTE] est [NOM COMPLET]."
- Si la question demande une procédure, donne les étapes présentes dans le contexte.
- Si la réponse n'apparaît pas dans le contexte, écris "Information non disponible."
- Reste concis et précis.

Réponse :"""

_GENERATION_PROMPT_BULLETED = """<contexte>
{context}
</contexte>

Sources du contexte : {sources}

{history}
<question>
{question}
</question>

INSTRUCTIONS :
- Donne une liste numérotée d'étapes courtes et précises, uniquement à partir des informations du contexte.
- Utilise un style clair et direct avec un seul point par ligne.
- Ne fournis pas d'explication générale ni de commentaire supplémentaire.
- Si aucun élément pertinent n'est présent dans le contexte, écris "Information non disponible."

Réponse :"""

_GENERATION_PROMPT_LIST = """<contexte>
{context}
</contexte>

Sources du contexte : {sources}

{history}
<question>
{question}
</question>

INSTRUCTIONS :
- Donne une liste numérotée, un élément par ligne, uniquement à partir des informations du contexte.
- Ne fournis pas d'explication ni de commentaire supplémentaire.
- Si aucun élément pertinent n'est présent dans le contexte, écris "Information non disponible."

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
    parts = []
    for c in chunks:
        filename = c["metadata"].get("filename", "?")
        chunk_index = c["metadata"].get("chunk_index")
        header = f"[source: {filename}]"
        if chunk_index is not None:
            header += f" [chunk {chunk_index}]"
        parts.append(f"{header}\n{c['content']}")
    return "\n\n---\n\n".join(parts)




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


def _safe_answer(answer: str) -> str:
    if not answer or not answer.strip():
        return "Information non disponible."
    normalized = answer.strip()
    if normalized.lower() in {"none", "n/a", "?", "information non disponible"}:
        return "Information non disponible."
    return normalized



def _build_structured_answer(raw_row: Dict[str, str], table: str, engine) -> str:
    """Formate une ligne SQL en réponse lisible — zéro mot-clé hardcodé.

    Utilise deux analyses statistiques sur les données réelles :
    1. get_primary_column() → colonne identifiant (ratio unique élevé)
    2. get_role_column()    → colonne catégorie/rôle (cardinalité intermédiaire)
    3. Fallback longueur   → valeur la plus longue si aucune colonne détectée
    """
    clean = {k: str(v).strip().rstrip(".") for k, v in raw_row.items() if v and str(v).strip()}
    if not clean:
        return "Information non disponible."

    primary_col = engine.get_primary_column(table)
    primary_val = clean.get(primary_col, "") if primary_col else ""

    role_col = engine.get_role_column(table)
    role_val = clean.get(role_col, "") if role_col else ""

    if primary_val and role_val and primary_val.lower() != role_val.lower():
        return f"{primary_val} est {role_val}."

    # Fallback : tri par longueur (libellé le plus long = nom principal)
    sorted_vals = sorted(clean.values(), key=len, reverse=True)
    if len(sorted_vals) >= 2 and sorted_vals[0].lower() != sorted_vals[1].lower():
        return f"{sorted_vals[0]} — {sorted_vals[1]}"
    if sorted_vals:
        return sorted_vals[0]
    return " | ".join(f"{k}: {v}" for k, v in clean.items())



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

    # Chemin B : QA structurée avec filtre → SQL + LLM
    if (
        not exhaustive
        and source
        and not schema.get(source, {}).get("is_doc")
        and filt
        and structured.has_table(source)
    ):
        return "structured_qa_path"


    # Chemin D : QA sur Excel sans filtre → keyword search SQL direct (sans LLM)
    # Contourne la distorsion des noms propres par les petits modèles.
    if (
        not exhaustive
        and source
        and not schema.get(source, {}).get("is_doc")
        and structured.has_table(source)
        and intent_data.get("intent") == "qa"
    ):
        return "structured_qa_direct_path"


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


    # ── 3a. structured_qa_direct_node ────────────────────────────────────
    # QA sur Excel sans filtre : keyword search DuckDB → formatage direct, sans LLM.

    def structured_qa_direct_node(state: GraphState) -> dict:
        intent_data       = state["intent_data"]
        resolved_question = state["resolved_question"]
        source            = intent_data.get("source")

        rows = structured.keyword_search(source, resolved_question, max_results=3)
        if not rows:
            return {
                "answer":      "Information non disponible.",
                "sources":     [],
                "chunks_used": 0,
                "path_taken":  "structured_qa_direct",
            }

        sources = list({r["metadata"].get("filename", "?") for r in rows})
        raw_row = rows[0]["metadata"].get("raw_row", {})
        answer  = (
            _build_structured_answer(raw_row, source, structured)
            if raw_row
            else rows[0]["content"].split("] ", 1)[-1]
        )
        logger.info("  ✅ Structured QA Direct: %d ligne(s) SQL → sans LLM", len(rows))
        return {
            "answer":      answer,
            "sources":     sources,
            "chunks_used": len(rows),
            "path_taken":  "structured_qa_direct",
        }

    # ── 3. exhaustive_node ────────────────────────────────────────────────

    def exhaustive_node(state: GraphState) -> dict:
        intent_data = state["intent_data"]
        source      = intent_data.get("source")
        column      = intent_data.get("column")
        filt        = intent_data.get("filter") or {}

        # Règle : si on a un filtre, on ignore la colonne spécifique et on retourne
        # la ligne complète (column=None). Cela évite de retourner juste la valeur du filtre.
        # Ex: "quels sont les formations obligatoires?" → filtre STATUT=Obligatoire
        #     Si column=STATUT, on retourne "Obligatoire" au lieu de la ligne complète.
        #     Avec column=None, on retourne "INTITULE_DE_LA_FORMATION: XYZ | STATUT: Obligatoire"
        query_column = None if filt else column

        direct = structured.list_values(
            table=source, column=query_column, filters=filt, distinct=True,
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

        # Même règle que exhaustive_node : si on a un filtre, ignorer la colonne
        # spécifique et retourner la ligne complète
        query_column = None if filt else column

        qa_rows      = structured.list_values(table=source, column=query_column, filters=filt, distinct=True)
        sql_warnings = list(structured.last_warnings)

        context      = "\n".join(r["content"] for r in qa_rows[:20])
        sources      = list({r["metadata"].get("filename", "?") for r in qa_rows})
        if not context.strip():
            return {
                "answer":      "Information non disponible.",
                "sources":     [],
                "chunks_used": 0,
                "warnings":    sql_warnings,
                "path_taken":  "structured_qa",
            }

        history_text = _format_history(history)
        use_bullets  = intent_data.get("intent") == "detail"
        template     = _GENERATION_PROMPT_BULLETED if use_bullets else _GENERATION_PROMPT
        prompt = template.format(
            context=context,
            sources=", ".join(sources) if sources else "—",
            question=resolved_question,
            history=history_text,
        )
        answer = _safe_answer(llm.generate(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        ))
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
        from ..retrieval.hybrid_search import reciprocal_rank_fusion

        intent_data  = state["intent_data"]
        exhaustive   = intent_data.get("exhaustive", False)
        search_query = state["resolved_question"]

        query_emb = embedder.embed_single(search_query)

        if exhaustive:
            k_dense  = min(config.max_chunks_exhaustive * 5, vector_store.count())
            k_sparse = config.max_chunks_exhaustive * 5
        else:
            k_dense  = config.top_k_dense
            k_sparse = config.top_k_sparse

        dense  = vector_store.search(query_emb, k=k_dense)
        sparse = bm25.search(search_query, k=k_sparse)
        hybrid = reciprocal_rank_fusion(dense, sparse, k=config.rrf_k)

        logger.info("  [retrieve] Dense: %d | Sparse: %d | RRF: %d",
                    len(dense), len(sparse), len(hybrid))
        return {
            "search_query":   search_query,
            "hybrid_results": hybrid,
        }

    # ── 6. rerank_node ───────────────────────────────────────────────────

    def rerank_node(state: GraphState) -> dict:
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
        intent_data       = state["intent_data"]
        exhaustive        = intent_data.get("exhaustive", False)
        resolved_question = state["resolved_question"]
        history           = state.get("history")
        filtered          = state["filtered_chunks"]

        context      = _format_context(filtered)
        sources      = list({c["metadata"].get("filename", "?") for c in filtered})
        if not context.strip():
            return {
                "context":    "",
                "answer":     "Information non disponible.",
                "path_taken": "semantic_rag",
            }

        history_text = _format_history(history)
        use_bullets = exhaustive or intent_data.get("intent") == "detail"
        template = _GENERATION_PROMPT_BULLETED if use_bullets else _GENERATION_PROMPT
        prompt = template.format(
            context=context,
            sources=", ".join(sources) if sources else "—",
            question=resolved_question,
            history=history_text,
        )
        max_tokens = config.llm_max_tokens_long if exhaustive else config.llm_max_tokens

        answer = _safe_answer(llm.generate(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=config.llm_temperature,
            max_tokens=max_tokens,
        ))
        return {
            "context":    context,
            "answer":     answer,
            "path_taken": "semantic_rag",
        }

    # ── 8. finalize_node ─────────────────────────────────────────────────

    def finalize_node(state: GraphState) -> dict:
        path = state.get("path_taken", "semantic_rag")
        if path == "semantic_rag":
            filtered = state.get("filtered_chunks", [])
            sources  = _extract_sources(filtered)
            chunks   = len(filtered)
        else:
            sources = state.get("sources", [])
            chunks  = state.get("chunks_used", 0)
        return {
            "sources":     sources,
            "chunks_used": chunks,
        }

    return {
        "contextualize_node":          contextualize_node,
        "intent_node":                 intent_node,
        "structured_qa_direct_node":   structured_qa_direct_node,
        "exhaustive_node":             exhaustive_node,
        "structured_qa_node":          structured_qa_node,
        "retrieve_node":               retrieve_node,
        "rerank_node":                 rerank_node,
        "generate_node":               generate_node,
        "finalize_node":               finalize_node,
    }
