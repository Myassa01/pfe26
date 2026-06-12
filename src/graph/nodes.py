"""
Nœuds LangGraph pour le pipeline RAG.

Chaque nœud : (state: GraphState) -> dict  (mise à jour partielle de l'état)
"""
import logging
import re
from typing import Any, Dict, List, Optional

from .state import GraphState
from ..security import redact_row, scrub_answer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Tu es un assistant RH de l'ENGTP. Réponds en français de façon claire, concise et factuelle.
Règles :
1. Utilise UNIQUEMENT le contexte fourni. Si l'information est absente → "Information non disponible."
2. N'invente rien. Ne génère pas d'informations supplémentaires absentes du contexte.
3. Ne réponds pas avec des procédures ou des détails génériques qui ne figurent pas explicitement dans le contexte.
4. Pour une question sur une personne ou un poste, réponds avec un énoncé court et précis.
5. Si la question porte sur les obligations de l'entreprise, liste UNIQUEMENT ce que le contexte mentionne explicitement.
6. Si tu ne peux pas vérifier la réponse dans le contexte, réponds "Information non disponible."."""

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
- Si la question demande une liste (formations, documents, étapes...), liste TOUS les éléments mentionnés dans le contexte.
- Si la question porte sur un lieu, une date, une certification ou un chiffre, extrais la valeur exacte du contexte.
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

# Mots grammaticaux universels — jamais présents comme données dans un fichier Excel.
# Pas de vocabulaire métier : fonctionne pour tout fichier, toute langue.
_STOP_WORDS: set = {
    # Français
    "quel", "quelle", "quels", "quelles", "sont", "est", "les", "des", "dans",
    "pour", "avec", "par", "sur", "une", "tous", "tout", "toutes", "liste",
    "donne", "donner", "affiche", "afficher", "montre", "montrer", "quels",
    "veux", "voulez", "donner", "avoir", "faire", "dire", "voir",
    # Anglais
    "what", "which", "show", "give", "list", "the", "all", "from", "have",
    "want", "tell", "make",
}


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
    """Normalise un texte pour comparaison insensible aux accents/casse/ponctuation.

    - Minuscules
    - Supprime apostrophes (droites et courbes), points, tirets
      → H.RMEL == H.R'MEL == HRMEL
    - Remplace les caractères accentués par leur équivalent ASCII
    """
    text = (
        text.lower()
        .replace("'", "")
        .replace("\u2019", "")   # apostrophe courbe
        .replace(".", "")
        .replace("-", "")
    )
    for src, dst in [
        ("\xe9", "e"), ("\xe8", "e"), ("\xea", "e"), ("\xeb", "e"),
        ("\xe0", "a"), ("\xe2", "a"), ("\xe4", "a"),
        ("\xee", "i"), ("\xef", "i"), ("\xf4", "o"), ("\xf6", "o"),
        ("\xf9", "u"), ("\xfb", "u"), ("\xfc", "u"), ("\xe7", "c"),
        ("\u0153", "oe"), ("\xe6", "ae"),
    ]:
        text = text.replace(src, dst)
    return text


def _sig_tokens(question: str) -> List[str]:
    """Extrait les tokens porteurs de sens d'une question.

    Règle : token normalisé par _fold, longueur ≥ 4, hors _STOP_WORDS.
    Aucun vocabulaire métier hardcodé — fonctionne pour tout domaine.

    Exemples :
      "quels sont les services de SKIKDA" → ["services", "skikda"]
      "quels sont les services de ARZEW"  → ["services", "arzew"]
      "liste des départements de H.RMEL"  → ["departements", "hrmel"]
    """
    tokens = [
        t for t in re.split(r"[\s\-_/,;:!?.]+", _fold(question))
        if len(t) >= 4 and t not in _STOP_WORDS
    ]
    return tokens


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


def _fold_attr(s: str) -> str:
    """Normalise une chaîne : majuscules, sans accents, sans espaces/underscores."""
    import unicodedata
    s = unicodedata.normalize("NFD", s.upper())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace(" ", "").replace("_", "")


_IDENTITY_MARKERS = [
    "qui est", "qui sont", "qui occupe", "qui dirige", "qui gere",
    "qui assure", "qui fait", "qui travaille", "who is", "who are",
]


def _is_identity_question(question: str) -> bool:
    q = question.lower().strip()
    return any(marker in q for marker in _IDENTITY_MARKERS)


def _is_attribute_question(question: str) -> bool:
    if _is_identity_question(question):
        return False
    q = _fold(question)
    if "combien" in q:
        return True
    if re.search(r"\bquel(?:le)?\s+est\b", q):
        return True
    return False


def _find_target_column(
    question: str,
    raw_row: Dict[str, str],
    primary_col: Optional[str] = None,
    role_col: Optional[str] = None,
    engine=None,
    table: Optional[str] = None,
) -> Optional[str]:
    if _is_identity_question(question):
        return None

    excluded = [c for c in (primary_col, role_col) if c]

    if engine and table and engine.has_table(table):
        col = engine.find_column_for_question(table, question, excluded_cols=excluded)
        if col:
            return col

    q_folded = _fold_attr(question)
    excluded_set = set(excluded)
    for col in raw_row:
        if col in excluded_set:
            continue
        cf = _fold_attr(col)
        if len(cf) >= 4 and cf in q_folded and raw_row.get(col, "").strip():
            return col

    return None


def _format_structured_answer(
    raw_row: Dict[str, str],
    table: Optional[str] = None,
    engine=None,
) -> str:
    def _up(v: str) -> str:
        return v.strip().upper()

    def _is_short_code(v: str) -> bool:
        s = v.strip()
        return len(s) <= 3 and s.upper() == s and s.replace("-", "").isalpha()

    all_cells = {k: str(v).strip() for k, v in raw_row.items() if v and str(v).strip()}
    if not all_cells:
        return " | ".join(f"{k}: {v}" for k, v in raw_row.items() if v)

    if engine and table and engine.has_table(table):
        primary_col = engine.get_primary_column(table)
        role_col    = engine.get_role_column(table)
        user_cols   = (engine.tables[table].get("user_columns")
                       or engine.tables[table]["columns"])

        primary_val = (all_cells.get(primary_col) or "").strip()
        role_val    = (all_cells.get(role_col)    or "").strip()

        name_val = primary_val
        if primary_val and len(primary_val) < 15 and not _is_short_code(primary_val):
            p_idx = user_cols.index(primary_col) if primary_col in user_cols else 99
            companion_col: Optional[str] = None
            best_score = 0.0

            for col in user_cols:
                if col in (primary_col, role_col):
                    continue
                val = (all_cells.get(col) or "").strip()
                if not val or _is_short_code(val) or len(val) > 20:
                    continue
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
                proximity = 1.0 / (1.0 + abs(c_idx - p_idx))
                brevity   = 1.0 if len(val) <= 20 else 0.4
                score     = proximity * brevity
                if score > best_score:
                    best_score    = score
                    companion_col = col

            if companion_col and best_score >= 0.25:
                companion_val = (all_cells.get(companion_col) or "").strip()
                c_idx = user_cols.index(companion_col) if companion_col in user_cols else 99
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

    # Fallback heuristique colonnes
    _nom_excl: set = set()
    for k in raw_row:
        for part in re.split(r"[_\s]+", k.upper().strip()):
            if len(part) >= 5 and part.isalpha():
                _nom_excl.add(part)

    nom = prenom = nom_complet = fonction = None

    for k, v in all_cells.items():
        ku     = k.upper().replace("_", "").replace(" ", "")
        ku_sep = k.upper().replace(" ", "_")

        if any(pat in ku for pat in ("NOMCOMPLET", "NOMETPRENOM", "NOMPRENOM",
                                     "FULLNAME", "NOMEMPLOY")):
            nom_complet = v
            continue
        if ku in ("PRENOM", "PRENOMS", "FIRSTNAME", "GIVENNAME"):
            prenom = v
            continue
        if ku_sep.startswith("PRENOM_") and "COMPLET" not in ku and "NOM" not in ku:
            prenom = v
            continue
        if ku in ("NOM", "LASTNAME", "FAMILYNAME", "SURNAME", "NAME"):
            nom = v
            continue
        if ku_sep.startswith("NOM_"):
            suffix = set(re.split(r"[_\s]+", ku_sep[4:]))
            if not suffix & _nom_excl:
                nom = v
            continue
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

            val_candidates = [
                v for k, v in pairs.items()
                if k != "__RAW__"
                and v.upper().startswith(table_stem + " ")
                and len(v) > len(table_stem)
            ]
            if val_candidates:
                return max(val_candidates, key=len)

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

    if exhaustive and source:
        if source in schema and not schema[source].get("is_doc"):
            if structured.has_table(source):
                return "exhaustive_path"

    resolved_q = state.get("resolved_question", "")
    if (
        not exhaustive
        and source
        and not schema.get(source, {}).get("is_doc")
        and filt
        and structured.has_table(source)
        and not _is_identity_question(resolved_q)
    ):
        return "structured_qa_path"

    if (
        not exhaustive
        and source
        and not schema.get(source, {}).get("is_doc")
        and structured.has_table(source)
        and intent_data.get("intent") == "qa"
    ):
        return "structured_qa_direct_path"

    if (
        not exhaustive
        and source is None
        and intent_data.get("intent") == "qa"
        and bool(structured.tables)
    ):
        return "structured_qa_direct_path"

    return "rag_path"


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY DE NŒUDS
# ─────────────────────────────────────────────────────────────────────────────

def build_nodes(components: Dict[str, Any]) -> Dict[str, Any]:
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

    # ── 3a. structured_qa_direct_node ─────────────────────────────────────
    # QA ponctuelle sur Excel : keyword search DuckDB → formatage direct, sans LLM.
    # Si DuckDB ne trouve rien → needs_rag_fallback=True → RAG + LLM prend le relais.

    def structured_qa_direct_node(state: GraphState) -> dict:
        intent_data       = state["intent_data"]
        resolved_question = state["resolved_question"]
        source            = intent_data.get("source")

        rows: list = []
        if source and structured.has_table(source):
            rows = structured.keyword_search(source, resolved_question, max_results=3)

        # Fallback multi-tables
        if not rows:
            best_rows: list = []
            best_table: Optional[str] = source
            best_is_and = False
            best_priority = 0  # tie-break déterministe : DIRECTION > DEPARTEMENT > SERVICE

            # Ordre de priorité des tables pour les questions hiérarchiques
            _TABLE_PRIORITY = {"DIRECTION": 3, "DEPARTEMENT": 2, "SERVICE": 1}

            # Détecte si la question porte sur un niveau hiérarchique précis
            _q_fold = _fold(resolved_question)
            _q_is_direction = bool(re.search(
                r"\b(directeur|directrice|direction|responsable\s+\w+\s+des\s+syst|"
                r"responsable\s+s[eé]curit[eé])\b", _q_fold
            ))
            _q_is_dept = bool(re.search(
                r"\b(departement|chef\s+du\s+dep|chef\s+de\s+dep)\b", _q_fold
            ))
            _q_is_service = bool(re.search(
                r"\b(service|chef\s+du\s+serv|chef\s+de\s+serv)\b", _q_fold
            ))

            for table_name in structured.tables:
                if table_name == source:
                    continue
                t_rows = structured.keyword_search(table_name, resolved_question, max_results=2)
                if not t_rows:
                    continue
                t_is_and = any(r["metadata"].get("and_match", False) for r in t_rows)
                t_priority = _TABLE_PRIORITY.get(table_name.upper(), 0)

                # Bonus de priorité si la table correspond exactement au niveau demandé
                if _q_is_direction and table_name.upper() == "DIRECTION":
                    t_priority += 10
                elif _q_is_dept and table_name.upper() == "DEPARTEMENT":
                    t_priority += 10
                elif _q_is_service and table_name.upper() == "SERVICE":
                    t_priority += 10

                # Règle de sélection : AND > OR, puis priorité table, puis volume
                better = False
                if t_is_and and not best_is_and:
                    better = True
                elif t_is_and == best_is_and:
                    if t_priority > best_priority:
                        better = True
                    elif t_priority == best_priority and len(t_rows) > len(best_rows):
                        better = True

                if better:
                    best_rows     = t_rows
                    best_table    = table_name
                    best_is_and   = t_is_and
                    best_priority = t_priority

            if best_rows:
                rows   = best_rows
                source = best_table
                logger.info("  [direct] Fallback multi-tables → %s (and=%s, priority=%d)",
                            source, best_is_and, best_priority)

        # Aucun résultat DuckDB → RAG + LLM
        if not rows:
            logger.info("  [direct] Aucun résultat Excel → fallback RAG+LLM")
            return {
                "needs_rag_fallback": True,
                "answer":             "",
                "sources":            [],
                "chunks_used":        0,
                "path_taken":         "structured_qa_direct",
            }

        # ── Validation du rôle : recouvrement de tokens (souple) ────────────
        # On ne rejette la ligne que si MOINS de 35 % des tokens significatifs
        # de la question se retrouvent dans la valeur de la colonne rôle.
        # Cela évite de rejeter "directeur central logistique" pour la question
        # "qui dirige la Direction Centrale Logistique" (fix S017).
        # Le seuil bas (0.35) est intentionnel : il élimine les vrais faux positifs
        # (ex: "ingenieur en chef HSE" pour "Chef du Département HSE") tout en
        # acceptant les variantes lexicales (directeur vs dirige, centrale vs central).
        if source and structured.has_table(source):
            role_col_chk = structured.get_role_column(source)
            if role_col_chk:
                raw_row_chk = rows[0]["metadata"].get("raw_row", {})
                role_found  = _fold(str(raw_row_chk.get(role_col_chk, "")))
                q_folded    = _fold(resolved_question)
                _ROLE_STOP  = {
                    "qui", "est", "sont", "quel", "quelle",
                    "le", "la", "les", "de", "du", "des", "un", "une",
                    "dans", "par", "pour", "sur", "avec",
                    "dirige", "responsable",
                }
                q_role_tokens = [
                    t for t in re.split(r"[\s\-_/,;:!?.]+", q_folded)
                    if len(t) >= 4 and t not in _ROLE_STOP
                ]
                role_tokens = set(re.split(r"[\s\-_/,;:!?.]+", role_found))

                if q_role_tokens and role_found:
                    # Recouvrement : token question préfixe/contenu dans rôle ou vice-versa
                    def _tok_match(qt: str, role_toks: set) -> bool:
                        for rt in role_toks:
                            if len(rt) < 3:
                                continue
                            if qt.startswith(rt) or rt.startswith(qt):
                                return True
                            # Tolérance 1 char (pluriel, féminin)
                            if qt[:-1] == rt[:-1] and len(qt) >= 5:
                                return True
                        return False

                    matched = sum(1 for t in q_role_tokens if _tok_match(t, role_tokens))
                    match_ratio = matched / len(q_role_tokens)

                    if match_ratio < 0.35:
                        logger.info(
                            "  [direct] Rôle '%s' ≠ question (match=%.0f%%) → fallback RAG+LLM",
                            role_found[:60], match_ratio * 100,
                        )
                        return {
                            "needs_rag_fallback": True,
                            "answer":             "",
                            "sources":            [],
                            "chunks_used":        0,
                            "path_taken":         "structured_qa_direct",
                        }

        sources = list({r["metadata"].get("filename", "?") for r in rows})
        raw_row = redact_row(rows[0]["metadata"].get("raw_row", {}), state.get("user_role"))
        if raw_row:
            primary_col = structured.get_primary_column(source) if source else None
            role_col    = structured.get_role_column(source)    if source else None
            target_col  = _find_target_column(
                resolved_question, raw_row, primary_col, role_col,
                engine=structured, table=source,
            )
            if target_col:
                target_val  = str(raw_row[target_col]).strip()
                primary_val = str(raw_row.get(primary_col, "")).strip() if primary_col else ""
                if primary_val and len(primary_val) < 15 and source and structured.has_table(source):
                    user_cols = (structured.tables[source].get("user_columns")
                                 or structured.tables[source]["columns"])
                    p_idx = user_cols.index(primary_col) if primary_col in user_cols else 99
                    for col in user_cols:
                        if col in (primary_col, role_col, target_col):
                            continue
                        v = str(raw_row.get(col, "")).strip()
                        if not v or len(v) > 20:
                            continue
                        c_idx = user_cols.index(col) if col in user_cols else 99
                        if abs(c_idx - p_idx) <= 2:
                            primary_val = f"{v} {primary_val}" if c_idx < p_idx else f"{primary_val} {v}"
                            break
                col_label = target_col.replace("_", " ").title()
                answer = f"{primary_val.upper()} — {col_label}: {target_val}" if primary_val else f"{col_label}: {target_val}"
            elif _is_attribute_question(resolved_question):
                # Attribut demandé mais colonne absente → RAG+LLM peut avoir la réponse
                logger.info("  [direct] Attribut absent de la table → fallback RAG+LLM")
                return {
                    "needs_rag_fallback": True,
                    "answer":             "",
                    "sources":            [],
                    "chunks_used":        0,
                    "path_taken":         "structured_qa_direct",
                }
            else:
                answer = _format_structured_answer(raw_row, table=source, engine=structured)
        else:
            answer = rows[0]["content"].split("] ", 1)[-1]

        logger.info("  ✅ Structured QA Direct → sans LLM [table=%s]", source)
        return {
            "answer":      answer,
            "sources":     sources,
            "chunks_used": len(rows),
            "path_taken":  "structured_qa_direct",
        }

    # ── 3b. exhaustive_node ───────────────────────────────────────────────
    # Liste exhaustive depuis Excel.
    # CORRECTIF FILTRAGE : extrait les tokens significatifs de la question
    # (≥ 4 chars, hors mots grammaticaux) pour faire un keyword_search ciblé
    # quand filt={}.  "quels sont les services de SKIKDA" → ["services","skikda"]
    # Aucun mot métier hardcodé — fonctionne pour tout fichier.

    def exhaustive_node(state: GraphState) -> dict:
        intent_data   = state["intent_data"]
        source        = intent_data.get("source")
        column        = intent_data.get("column")
        filt          = intent_data.get("filter") or {}
        resolved_q    = state.get("resolved_question", "")

        query_column = column if column else structured.get_entity_column(source)

        # ── Étape 1 : keyword_search sur tokens significatifs si filt vide ──
        # Permet de filtrer "services de SKIKDA" sans que l'intent_router
        # ait extrait SKIKDA comme filtre explicite.
        if not filt and resolved_q:
            tokens   = _sig_tokens(resolved_q)
            search_q = " ".join(tokens) if tokens else resolved_q
            logger.info("  [exhaustive] filt={} → tokens: %s", tokens)

            kw_rows  = structured.keyword_search(source, search_q, max_results=200)
            and_rows = [r for r in kw_rows if r["metadata"].get("and_match", False)]

            if and_rows:
                seen_kw: set  = set()
                unique_kw: list = []
                for r in and_rows:
                    raw_row = r["metadata"].get("raw_row", {})
                    val = (
                        str(raw_row.get(query_column, "")).strip()
                        if query_column and raw_row
                        else r["content"].split("] ", 1)[-1]
                    )
                    key = _fold(val)
                    if key and key not in seen_kw and len(val) > 2:
                        seen_kw.add(key)
                        unique_kw.append(val)
                if unique_kw:
                    sources_kw = list({r["metadata"].get("filename", "?") for r in and_rows})
                    _cnt_p = [r"combien", r"quel nombre", r"nombre de"]
                    _is_cnt = any(re.search(p, resolved_q.lower()) for p in _cnt_p) if resolved_q else False
                    if _is_cnt:
                        body = f"Il y a {len(unique_kw)} éléments."
                    else:
                        body = (
                            f"Il y a {len(unique_kw)} résultat(s) :\n"
                            + "\n".join(f"{i+1}. {n}" for i, n in enumerate(unique_kw))
                        )
                    logger.info("  ✅ Exhaustive (keyword AND): %d éléments", len(unique_kw))
                    logger.info("  ✅ Exhaustive (keyword AND): %d éléments", len(unique_kw))
                    return {
                        "answer":      body,
                        "sources":     sources_kw,
                        "chunks_used": len(unique_kw),
                        "warnings":    [],
                        "path_taken":  "exhaustive",
                    }

            # Si kw_rows vide (ex: "disponible" n'est pas une valeur de colonne),
            # on laisse passer à l'Étape 2 qui liste TOUS les éléments de la table.
            # Le fallback RAG est géré plus bas si list_values retourne aussi rien.

        # ── Étape 2 : list_values avec filtre SQL explicite ───────────────
        direct       = structured.list_values(table=source, column=query_column, filters=filt, distinct=True)
        sql_warnings = list(structured.last_warnings)

        # Fallback filtre spéciaux (apostrophe, point…)
        filter_has_special = bool(filt and any(
            re.search(r"[.\'\-]", str(v)) for v in filt.values()
        ))
        if (not direct or (filter_has_special and len(direct) < 5)) and filt:
            filter_vals = " ".join(str(v) for v in filt.values() if v)
            if filter_vals:
                kw_rows = structured.keyword_search(source, filter_vals, max_results=50)
                if kw_rows:
                    entity_col  = query_column
                    seen_fb: set = {_fold(d["content"]) for d in direct}
                    fb_extra: list = []
                    for kw in kw_rows:
                        raw_row = kw["metadata"].get("raw_row", {})
                        val = str(raw_row.get(entity_col, "")).strip() if entity_col and raw_row else ""
                        if not val:
                            val = kw["content"]
                        key = _fold(val)
                        if key and key not in seen_fb:
                            seen_fb.add(key)
                            fb_extra.append({"content": val, "metadata": kw["metadata"]})
                    direct = direct + fb_extra

        # Aucun résultat → RAG+LLM
        if not direct:
            logger.info("  [exhaustive] Aucun résultat Excel → fallback RAG+LLM")
            return {
                "needs_rag_fallback": True,
                "answer":             "",
                "sources":            [],
                "chunks_used":        0,
                "warnings":           sql_warnings,
                "path_taken":         "exhaustive",
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

        # Nettoyage stem
        source_stem = source.upper() if source else ""
        if source_stem and len(source_stem) >= 5:
            processed: list = []
            for name in unique_names:
                idx = name.upper().find(source_stem)
                processed.append(name[idx:].strip() if idx > 0 else name)
            seen2: set = set()
            clean: list = []
            for n in processed:
                k = _fold(n)
                if k not in seen2:
                    seen2.add(k)
                    clean.append(n)
            unique_names = clean

        # Suppression artefacts Excel
        artifacts_folded: set = set()
        if filt:
            for v in filt.values():
                artifacts_folded.add(_fold(str(v).strip()))
        if query_column:
            artifacts_folded.add(_fold(query_column.strip()))
            artifacts_folded.add(_fold(query_column.strip().replace("_", " ")))
        if artifacts_folded:
            unique_names = [n for n in unique_names if _fold(n.strip()) not in artifacts_folded]

        prefix_lines = []
        if sql_warnings:
            prefix_lines.append("⚠ Note :")
            for w in sql_warnings:
                prefix_lines.append(f"  • {w}")
            prefix_lines.append("")

        # ── Détection question de comptage ──────────────────────────────────
        _counting_patterns = [r'combien', r'quel nombre', r'nombre de']
        _is_counting = any(re.search(p, resolved_q.lower()) for p in _counting_patterns) if resolved_q else False

        if _is_counting:
            body = f"Il y a {len(unique_names)} éléments."
        else:
            body = f"Il y a {len(unique_names)} résultat(s) :\n" + "\n".join(f"{i+1}. {n}" for i, n in enumerate(unique_names))
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

        query_column = None if filt else column
        qa_rows      = structured.list_values(table=source, column=query_column, filters=filt, distinct=True)
        sql_warnings = list(structured.last_warnings)

        context = "\n".join(r["content"] for r in qa_rows[:20])
        sources = list({r["metadata"].get("filename", "?") for r in qa_rows})

        # Aucun résultat Excel → RAG+LLM
        if not context.strip():
            logger.info("  [structured_qa] Aucun résultat Excel → fallback RAG+LLM")
            return {
                "needs_rag_fallback": True,
                "answer":             "",
                "sources":            [],
                "chunks_used":        0,
                "warnings":           sql_warnings,
                "path_taken":         "structured_qa",
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
        elif intent in ("list", "detail"):
            reranked = reranker.rerank(
                query=search_query,
                documents=hybrid[:30],
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

        context = _format_context(filtered)
        sources = list({c["metadata"].get("filename", "?") for c in filtered})
        if not context.strip():
            return {
                "context":    "",
                "answer":     "Information non disponible.",
                "path_taken": "semantic_rag",
            }

        history_text = _format_history(history)
        use_bullets  = exhaustive or intent_data.get("intent") == "detail"
        template     = _GENERATION_PROMPT_BULLETED if use_bullets else _GENERATION_PROMPT
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

        # ── Garde-fou de confidentialité (niveau 2 : la réponse) ─────────────
        # Filet de sécurité : masque toute paire « libellé sensible : valeur »
        # ayant pu subsister dans la réponse d'un utilisateur non privilégié.
        out: dict = {"sources": sources, "chunks_used": chunks}
        role   = state.get("user_role")
        answer = state.get("answer", "")
        scrubbed = scrub_answer(answer, role)
        if scrubbed != answer:
            logger.info("  [sécurité] Réponse filtrée (role=%s)", role)
            out["answer"] = scrubbed
        return out

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
