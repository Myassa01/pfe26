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




def _format_structured_answer(
    raw_row: Dict[str, str],
    table: Optional[str] = None,
    engine=None,
) -> str:
    """Formate une ligne Excel en réponse courte, sans LLM.

    CHEMIN 1 — engine disponible (prioritaire) :
      Détection 100 % statistique via DuckDB, aucun mot-clé hardcodé.
      - get_primary_column() → identifie le champ principal (nom, identifiant…)
        en se basant sur l'unicité et la longueur moyenne des valeurs.
      - get_role_column()    → identifie le champ rôle/fonction en se basant
        sur la cardinalité et la longueur des valeurs.
      - Pour les noms split NOM + PRENOM : cherche la colonne "compagne"
        (haute unicité, valeur courte, proche dans l'ordre des colonnes).

    CHEMIN 2 — sans engine (fallback uniquement) :
      Heuristiques sur les noms de colonnes. Utilisé seulement si DuckDB
      n'est pas disponible (ex : appel depuis un contexte hors pipeline).
    """
    def _up(v: str) -> str:
        return v.strip().upper()

    def _is_short_code(v: str) -> bool:
        s = v.strip()
        return len(s) <= 3 and s.upper() == s and s.replace("-", "").isalpha()

    all_cells = {k: str(v).strip() for k, v in raw_row.items() if v and str(v).strip()}
    if not all_cells:
        return " | ".join(f"{k}: {v}" for k, v in raw_row.items() if v)

    # ══════════════════════════════════════════════════════════════════════
    # CHEMIN 1 : détection statistique (engine disponible)
    # ══════════════════════════════════════════════════════════════════════
    if engine and table and engine.has_table(table):
        primary_col = engine.get_primary_column(table)
        role_col    = engine.get_role_column(table)
        user_cols   = (engine.tables[table].get("user_columns")
                       or engine.tables[table]["columns"])

        primary_val = (all_cells.get(primary_col) or "").strip()
        role_val    = (all_cells.get(role_col)    or "").strip()

        # ── Détection NOM + PRENOM : si la valeur principale est courte,
        #    cherche une colonne "compagne" (haute unicité, valeur courte).
        #    Aucun nom de colonne hardcodé — on compare uniquement les stats.
        name_val = primary_val
        if primary_val and len(primary_val) < 15 and not _is_short_code(primary_val):
            p_idx = user_cols.index(primary_col) if primary_col in user_cols else 99
            companion_col: Optional[str] = None
            best_score = 0.0

            for col in user_cols:
                if col in (primary_col, role_col):
                    continue
                val = (all_cells.get(col) or "").strip()
                # Valeur doit être non-vide, courte (prénom/nom), pas un code
                if not val or _is_short_code(val) or len(val) > 20:
                    continue
                # Exclure les colonnes statut/flag : très peu de valeurs distinctes
                # (ex : CONFIRME avec 1 seule valeur, OUI/NON avec 2 valeurs).
                if engine and table and engine.has_table(table):
                    try:
                        _st = engine.tables[table]["sql_table"]
                        _nr = engine.tables[table]["row_count"]
                        _nd = engine.conn.execute(
                            f'SELECT COUNT(DISTINCT "{col}") FROM "{_st}" '
                            f'WHERE "{col}" IS NOT NULL AND TRIM("{col}") <> \'\''
                        ).fetchone()[0]
                        if _nr >= 4 and _nd <= 2:
                            continue
                    except Exception:
                        pass
                c_idx     = user_cols.index(col) if col in user_cols else 99
                # Score = proximité dans l'ordre des colonnes × brièveté de la valeur
                proximity = 1.0 / (1.0 + abs(c_idx - p_idx))
                brevity   = 1.0 if len(val) <= 20 else 0.4
                score     = proximity * brevity
                if score > best_score:
                    best_score    = score
                    companion_col = col

            if companion_col and best_score >= 0.25:
                companion_val = (all_cells.get(companion_col) or "").strip()
                c_idx = user_cols.index(companion_col) if companion_col in user_cols else 99
                # Respecte l'ordre des colonnes pour PRENOM NOM ou NOM PRENOM
                if c_idx < p_idx:
                    name_val = f"{companion_val} {primary_val}"
                else:
                    name_val = f"{primary_val} {companion_val}"

        parts: List[str] = []
        if name_val and not _is_short_code(name_val):
            parts.append(_up(name_val))
        if role_val and not _is_short_code(role_val):
            parts.append(_up(role_val))
        if parts:
            return " — ".join(parts)

    # ══════════════════════════════════════════════════════════════════════
    # CHEMIN 2 : heuristiques colonnes (fallback sans engine)
    # ══════════════════════════════════════════════════════════════════════

    # Exclusion NOM_* construite depuis les noms de colonnes présents
    _nom_excl: set = set()
    for k in raw_row:
        for part in re.split(r"[_\s]+", k.upper().strip()):
            if len(part) >= 5 and part.isalpha():
                _nom_excl.add(part)

    nom = prenom = nom_complet = fonction = None

    for k, v in all_cells.items():
        ku     = k.upper().replace("_", "").replace(" ", "")
        ku_sep = k.upper().replace(" ", "_")

        # Nom complet
        if any(pat in ku for pat in ("NOMCOMPLET", "NOMETPRENOM", "NOMPRENOM",
                                     "FULLNAME", "NOMEMPLOY")):
            nom_complet = v
            continue
        # Prénom
        if ku in ("PRENOM", "PRENOMS", "FIRSTNAME", "GIVENNAME"):
            prenom = v
            continue
        if ku_sep.startswith("PRENOM_") and "COMPLET" not in ku and "NOM" not in ku:
            prenom = v
            continue
        # Nom de famille
        if ku in ("NOM", "LASTNAME", "FAMILYNAME", "SURNAME", "NAME"):
            nom = v
            continue
        if ku_sep.startswith("NOM_"):
            suffix = set(re.split(r"[_\s]+", ku_sep[4:]))
            if not suffix & _nom_excl:
                nom = v
            continue
        # Fonction / rôle
        col_parts = set(re.split(r"[_\s]+", k.upper().strip()))
        if col_parts & {"FONCTION", "POSTE", "EMPLOI", "TITRE", "GRADE",
                        "QUALIF", "FUNCTION", "POSITION", "ROLE", "TITLE", "JOB"}:
            if not fonction or len(v) > len(fonction):
                fonction = v

    parts = []
    if nom_complet:
        parts.append(_up(nom_complet))
    elif prenom and nom:
        parts.append(f"{_up(prenom)} {_up(nom)}")
    elif nom:
        parts.append(_up(nom))
    elif prenom:
        parts.append(_up(prenom))
    if fonction:
        parts.append(_up(fonction))
    if parts:
        return " — ".join(parts)

    # Dernier recours : 3 valeurs les plus longues
    long_vals = [v for v in all_cells.values() if not _is_short_code(v)]
    return " — ".join(sorted(long_vals, key=len, reverse=True)[:3])


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
        # Priority 1 : colonne dont le nom contient le stem de la table
        # → colonne entité (NOM_DEPARTEMENT, DIRECTION, NOM_SERVICE…).
        # En cas de plusieurs candidats, préfère la valeur la plus longue
        # (les noms d'entités sont plus longs que les noms de personnes).
        table_stem = source.upper().replace("_", "")
        if len(table_stem) >= 5:
            entity_candidates = [
                v for k, v in pairs.items()
                if k != "__RAW__"
                and (table_stem in k.replace("_", "") or k.replace("_", "") in table_stem)
                and len(v) > 3
            ]
            if entity_candidates:
                return max(entity_candidates, key=len)

    # Fallback : valeur la plus longue = nom d'entité (SERVICE MECANIQUE > KEDDAR)
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


    # Chemin D : QA sur Excel avec source identifiée → keyword search SQL direct (sans LLM)
    # Contourne la distorsion des noms propres par les petits modèles.
    if (
        not exhaustive
        and source
        and not schema.get(source, {}).get("is_doc")
        and structured.has_table(source)
        and intent_data.get("intent") == "qa"
    ):
        return "structured_qa_direct_path"

    # Chemin D' : QA sans source identifiée → essaie toutes les tables Excel.
    # Le nœud structured_qa_direct gère la recherche multi-tables et le fallback RAG.
    if (
        not exhaustive
        and source is None
        and intent_data.get("intent") == "qa"
        and bool(structured.tables)
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

        # ── Recherche principale dans la table identifiée ─────────────────
        rows: list = []
        if source and structured.has_table(source):
            rows = structured.keyword_search(source, resolved_question, max_results=3)

        # ── Fallback multi-tables si source vide ou aucun résultat ────────
        # Cherche dans toutes les tables Excel et retourne la plus pertinente.
        # Priorité : AND match (tous les tokens présents) > OR match (partiel).
        # Ex : "chef departement technique" → DEPARTEMENT (AND) gagne sur
        #      FORMATION (OR sur "technique" seulement).
        if not rows:
            best_rows: list = []
            best_table: Optional[str] = source
            best_is_and = False
            for table_name in structured.tables:
                if table_name == source:
                    continue
                t_rows = structured.keyword_search(table_name, resolved_question, max_results=2)
                if not t_rows:
                    continue
                t_is_and = any(r["metadata"].get("and_match", False) for r in t_rows)
                # AND match bat toujours un OR match ; à égalité, plus de résultats gagne
                if (t_is_and and not best_is_and) or \
                   (t_is_and == best_is_and and len(t_rows) > len(best_rows)):
                    best_rows  = t_rows
                    best_table = table_name
                    best_is_and = t_is_and
            if best_rows:
                rows   = best_rows
                source = best_table
                logger.info("  [direct] Source inconnue → table trouvée par fallback: %s (and=%s)",
                            source, best_is_and)

        if not rows:
            # Aucun résultat dans aucune table Excel → demande un fallback vers RAG.
            # La réponse peut se trouver dans un PDF/DOCX.
            logger.info("  [direct] Aucun résultat dans les tables Excel → fallback RAG")
            return {
                "needs_rag_fallback": True,
                "answer":             "",
                "sources":            [],
                "chunks_used":        0,
                "path_taken":         "structured_qa_direct",
            }

        sources = list({r["metadata"].get("filename", "?") for r in rows})
        raw_row = rows[0]["metadata"].get("raw_row", {})
        if raw_row:
            # Passe table + engine pour la détection statistique des colonnes
            answer = _format_structured_answer(raw_row, table=source, engine=structured)
        else:
            answer = rows[0]["content"].split("] ", 1)[-1]
        logger.info("  ✅ Structured QA Direct: %d ligne(s) SQL → sans LLM [table=%s]",
                    len(rows), source)
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

        # ── Nettoyage post-extraction ─────────────────────────────────────
        # Si la valeur contient le stem de la table APRÈS un préfixe
        # (ex : "CHEF DE DEPARTEMENT TECHNIQUE" → "DEPARTEMENT TECHNIQUE"),
        # on supprime ce préfixe. Les entrées sans le stem sont conservées
        # telles quelles (ex : "RESPONSABLE SECURITE" dans DIRECTION).
        source_stem = source.upper() if source else ""
        if source_stem and len(source_stem) >= 5:
            processed: list = []
            for name in unique_names:
                idx = name.upper().find(source_stem)
                if idx > 0:
                    processed.append(name[idx:].strip())   # retire le préfixe
                else:
                    processed.append(name)   # déjà propre (idx=0) ou sans stem (idx=-1)
            # Re-déduplique après transformation (deux préfixes différents → même entité)
            seen2: set = set()
            clean: list = []
            for n in processed:
                k = _fold(n)
                if k not in seen2:
                    seen2.add(k)
                    clean.append(n)
            unique_names = clean

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
