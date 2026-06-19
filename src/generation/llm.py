"""
Module d'analyse de CV via pipeline RAG.
Extraction CV + recherche exigences + analyse LLM.

Format de sortie LLM : JSON strict (beaucoup plus fiable à respecter pour un
petit modèle comme Qwen2.5-1.5B-Instruct qu'un format markdown à 6 sections).
Le backend reconstruit ensuite le texte markdown **SCORE**/**DOMAINE**/etc.
attendu par le frontend — aucun changement côté React n'est nécessaire.
Si le modèle ne produit pas de JSON valide, on retente une fois, puis on
retombe sur l'ancien parsing par regex sur texte libre en dernier recours.
"""

import io
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Extraction texte
# ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        logger.warning("PyMuPDF absent, fallback pdfplumber")
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                ).strip()
        except ImportError:
            raise RuntimeError("Aucune librairie PDF. Installez : pip install pymupdf")


def extract_text_from_docx(content: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except ImportError:
        raise RuntimeError("python-docx non installé. Installez : pip install python-docx")


def extract_cv_text(content: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif name.endswith(".docx"):
        return extract_text_from_docx(content)
    elif name.endswith(".txt"):
        return content.decode("utf-8", errors="replace").strip()
    raise ValueError(f"Format non supporté : {filename}. Formats acceptés : PDF, DOCX, TXT")


# ─────────────────────────────────────────────────────────────
# Validation CV
# ─────────────────────────────────────────────────────────────

MIN_CV_LENGTH = 200


def validate_cv_text(cv_text: str, filename: str) -> Optional[dict]:
    if not cv_text or len(cv_text.strip()) < MIN_CV_LENGTH:
        return {
            "answer": (
                f"⚠️ CV insuffisant ({filename}) — contenu trop limité.\n\n"
                f"Le fichier ne contient que {len(cv_text.strip())} caractères "
                f"(minimum requis : {MIN_CV_LENGTH}).\n\n"
                "Veuillez soumettre un CV complet incluant : formation, expériences et compétences."
            ),
            "score": 0,
            "poste": "Non analysé",
            "recommended_poste": "Non déterminable — CV incomplet",
            "sources": [],
            "elapsed_seconds": 0.0,
            "years_experience": None,
            "diploma_year": None,
            "filename": filename,
        }
    return None


# ─────────────────────────────────────────────────────────────
# Prompt — version JSON strict
# ─────────────────────────────────────────────────────────────
#
# Pourquoi JSON plutôt que markdown à sections ?
# Un petit modèle (1.5B) "dérive" facilement sur un format markdown à
# plusieurs sections imbriquées (il oublie une section, écrit de la prose
# libre, etc.). Un objet JSON plat avec des clés fixes est beaucoup plus
# facile à respecter pour un petit modèle ET beaucoup plus simple/robuste
# à parser côté backend (json.loads au lieu de regex fragiles).

ANALYSIS_PROMPT = """
Tu es un système ATS (Applicant Tracking System) expert en recrutement chez Sonatrach.

════════════════════════════════════════
POSTE CIBLE
════════════════════════════════════════
{poste}

════════════════════════════════════════
RÉFÉRENTIEL DES POSTES SONATRACH
════════════════════════════════════════
{job_context}

════════════════════════════════════════
CV DU CANDIDAT
════════════════════════════════════════
{cv_text}

════════════════════════════════════════
INSTRUCTIONS
════════════════════════════════════════

1. Identifie le domaine professionnel RÉEL du candidat (ex: Soudage, Comptabilité,
   Informatique, Espaces verts…), son diplôme le plus élevé et son année,
   ses années d'expérience totales dans son domaine, ses compétences principales.

2. Compare ce domaine au POSTE CIBLE :
   - Si les domaines sont totalement incompatibles (ex: jardinier vs comptable,
     soudeur vs développeur), le score DOIT être entre 0 et 2 sur 10, SANS EXCEPTION,
     même si le CV est excellent dans son propre domaine.
   - L'expérience Oil & Gas / Sonatrach n'est un bonus QUE si elle est dans le
     MÊME domaine que le poste cible. Sinon elle ne compte pas comme bonus.
   - Si POSTE CIBLE est "Non précisé", cherche dans le RÉFÉRENTIEL le poste qui
     correspond le mieux au domaine réel du candidat, et utilise son titre exact.
     Si rien ne correspond dans le référentiel, choisis un titre de poste court
     et naturel correspondant au vrai métier du candidat.

3. Score sur 10 (uniquement si compatibilité de domaine) :
   - Diplôme/formation pertinents pour le poste (0-3)
   - Compétences techniques spécifiques au poste (0-3)
   - Expérience professionnelle pertinente (0-3)
   - Expérience Sonatrach/Oil & Gas dans le MÊME domaine (0-1)

════════════════════════════════════════
FORMAT DE RÉPONSE — OBLIGATOIRE
════════════════════════════════════════

Réponds UNIQUEMENT avec un objet JSON valide. RIEN D'AUTRE.
Pas de markdown, pas de ```json, pas de texte avant ou après, pas d'explication.
Le JSON doit avoir EXACTEMENT ces clés :

{{
  "score": <entier 0 à 10>,
  "domaine": "Compatible" | "Partiellement compatible" | "Incompatible",
  "decision": "Recommandé" | "À étudier" | "Non recommandé",
  "atouts": ["point concis 1", "point concis 2", "point concis 3"],
  "lacunes": ["point concis 1", "point concis 2"],
  "poste_recommande": "titre court de 2 à 6 mots, JAMAIS une phrase",
  "annees_experience": <entier, -1 si inconnu>,
  "annee_diplome": <entier ex: 2013, 0 si inconnue>
}}

Exemple de poste_recommande VALIDE : "Comptable Principal", "Soudeur Qualifié Pipeline",
"Jardinier Qualifié". INVALIDE : une phrase explicative.
"""

RETRY_REMINDER = (
    "\n\nRAPPEL CRITIQUE : ta dernière réponse n'était pas un JSON valide. "
    "Réponds CETTE FOIS uniquement avec l'objet JSON demandé, rien d'autre, "
    "pas de texte avant ou après, pas de ```."
)


def build_analysis_prompt(cv_text: str, poste: str, job_context: str, retry: bool = False) -> str:
    poste_label = poste if poste else "Non précisé"
    prompt = ANALYSIS_PROMPT.format(
        poste=poste_label,
        job_context=job_context or (
            "Aucun référentiel disponible. "
            "Utilise les titres de postes courants dans l'industrie pétrolière algérienne."
        ),
        cv_text=cv_text[:4000],
    )
    if retry:
        prompt += RETRY_REMINDER
    return prompt


SYSTEM_PROMPT = (
    "Tu es un système ATS RH chez Sonatrach. Réponds UNIQUEMENT en français. "
    "Réponds UNIQUEMENT avec un objet JSON valide respectant exactement le schéma "
    "demandé. JAMAIS de texte hors du JSON, JAMAIS de ```json, JAMAIS d'explication. "
    "Sois strict et objectif : un profil hors-domaine (ex: jardinier postulant "
    "comptable) = score 0 à 2/10, incompatible, SANS EXCEPTION. "
    "Un profil dans le bon domaine avec une expérience solide = score élevé 7-9/10. "
    "Le champ poste_recommande doit TOUJOURS être un titre court de 2 à 6 mots, "
    "jamais une phrase."
)


# ─────────────────────────────────────────────────────────────
# Parsing JSON robuste
# ─────────────────────────────────────────────────────────────

def _extract_json_object(text: str) -> Optional[dict]:
    """Extrait et parse le premier objet JSON valide trouvé dans le texte.
    Tolère du texte parasite avant/après (préambule, ```json, etc.)."""
    if not text:
        return None

    # Tentative directe
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Tentative sur le bloc { ... } le plus large trouvé
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            # Nettoyage léger : virgules traînantes, guillemets simples
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            cleaned = cleaned.replace("'", '"')
            try:
                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def _is_sentence_not_title(value: str) -> bool:
    """Détecte si une valeur ressemble à une phrase plutôt qu'à un titre court."""
    if not value:
        return True
    v = value.strip()
    if len(v.split()) > 7:
        return True
    if v.endswith((":", ".", ",", ";")):
        return True
    sentence_markers = (
        "pour un candidat", "il serait", "il est conseillé", "nous recommandons",
        "tel que", "tels que", "serait conseillé", "serait préférable",
    )
    v_lower = v.lower()
    return any(marker in v_lower for marker in sentence_markers)


def _normalize_json_result(data: dict) -> Optional[dict]:
    """Valide et normalise un dict JSON parsé en champs propres et sûrs."""
    if not isinstance(data, dict):
        return None

    def _safe_int(value, lo, hi, default=None):
        try:
            v = int(value)
            return v if lo <= v <= hi else default
        except (TypeError, ValueError):
            return default

    score = _safe_int(data.get("score"), 0, 10)
    if score is None:
        return None  # le score est la seule clé vraiment indispensable

    domaine  = str(data.get("domaine") or "").strip() or None
    decision = str(data.get("decision") or "").strip() or None

    atouts = [str(a).strip() for a in (data.get("atouts") or []) if str(a).strip()]
    lacunes = [str(a).strip() for a in (data.get("lacunes") or []) if str(a).strip()]

    poste_rec = str(data.get("poste_recommande") or "").strip() or None
    if poste_rec and _is_sentence_not_title(poste_rec):
        poste_rec = None

    years_exp = _safe_int(data.get("annees_experience"), -1, 50)
    diploma_year = _safe_int(data.get("annee_diplome"), 1950, 2100)
    if diploma_year == 0:
        diploma_year = None

    return {
        "score": score,
        "domaine": domaine,
        "decision": decision,
        "atouts": atouts,
        "lacunes": lacunes,
        "poste_recommande": poste_rec,
        "annees_experience": years_exp,
        "annee_diplome": diploma_year,
    }


def _render_markdown_from_json(data: dict) -> str:
    """Reconstruit le texte au format markdown attendu par le frontend
    (**SCORE**, **DOMAINE**, etc.) à partir du JSON normalisé. Ainsi le
    frontend React n'a besoin d'aucune modification."""
    lines = [
        f"**SCORE** : {data['score']}/10",
    ]
    if data["domaine"]:
        lines.append(f"**DOMAINE** : {data['domaine']}")
    if data["decision"]:
        lines.append(f"**DÉCISION** : {data['decision']}")

    lines.append("")
    if data["atouts"]:
        lines.append("**ATOUTS**")
        for a in data["atouts"]:
            lines.append(f"- {a}")
        lines.append("")

    if data["lacunes"]:
        lines.append("**LACUNES**")
        for l in data["lacunes"]:
            lines.append(f"- {l}")
        lines.append("")

    if data["poste_recommande"]:
        lines.append(f"**POSTE RECOMMANDÉ** : {data['poste_recommande']}")

    years = data["annees_experience"]
    lines.append(f"**ANNÉES_EXPÉRIENCE** : {years if years is not None else -1}")

    diploma = data["annee_diplome"]
    lines.append(f"**ANNÉE_DIPLOME** : {diploma if diploma is not None else 0}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Analyse principale
# ─────────────────────────────────────────────────────────────

def analyze_cv_with_pipeline(
    pipeline, cv_text: str, poste: str, filename: str = "CV"
) -> dict:
    import time
    t0 = time.time()

    validation_error = validate_cv_text(cv_text, filename)
    if validation_error:
        logger.warning("CV '%s' rejeté : contenu insuffisant (%d car.)", filename, len(cv_text.strip()))
        return validation_error

    search_poste = poste.strip() if poste else ""

    # ── Requête RAG ────────────────────────────────────────────
    if search_poste:
        search_query = f"exigences diplômes compétences requis poste {search_poste} Sonatrach"
    else:
        cv_keywords = " ".join(cv_text[:800].split())[:400]
        search_query = f"postes Sonatrach référentiel {cv_keywords}"

    try:
        query_embedding = pipeline.embedder.embed_single(search_query)
        dense_results = pipeline.vector_store.search(query_embedding, k=pipeline.config.top_k_dense)

        sparse_results = []
        if pipeline.bm25:
            sparse_results = pipeline.bm25.search(search_query, k=pipeline.config.top_k_sparse)

        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=pipeline.config.rrf_k)
        top_chunks = fused[:pipeline.config.top_k_after_rerank]

        if pipeline.reranker and top_chunks:
            pairs = [(search_query, c["content"]) for c in top_chunks]
            scores = pipeline.reranker.model.predict(pairs)
            for c, s in zip(top_chunks, scores):
                c["rerank_score"] = float(s)
            top_chunks = sorted(top_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

        job_context = "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', '?')}]\n{c['content']}" for c in top_chunks
        )
        sources = list({c["metadata"].get("source", "?") for c in top_chunks})

    except Exception as e:
        logger.warning("Erreur RAG : %s", e)
        job_context = ""
        sources = []

    # ── Appel LLM (JSON strict, avec un retry si parsing échoue) ──────────
    raw_answer = None
    parsed = None

    for attempt in range(2):
        prompt = build_analysis_prompt(
            cv_text=cv_text, poste=search_poste, job_context=job_context,
            retry=(attempt == 1),
        )
        try:
            raw_answer = pipeline.llm.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=pipeline.config.llm_max_tokens_long,
            )
        except Exception as e:
            raise RuntimeError(f"Erreur LLM : {e}")

        json_data = _extract_json_object(raw_answer)
        parsed = _normalize_json_result(json_data) if json_data else None
        if parsed:
            break
        logger.warning(
            "Tentative %d : JSON invalide pour '%s'. Réponse brute :\n%s",
            attempt + 1, filename, (raw_answer or "")[:500],
        )

    if parsed:
        answer = _render_markdown_from_json(parsed)
        score = parsed["score"]
        recommended_poste = parsed["poste_recommande"]
        years_experience = parsed["annees_experience"]
        diploma_year = parsed["annee_diplome"]
    else:
        # ── Dernier recours : ancien parsing regex sur texte libre ────────
        logger.warning(
            "JSON non récupérable après 2 tentatives pour '%s' — fallback regex sur texte libre.",
            filename,
        )
        answer = raw_answer or ""
        score = _extract_score(answer)
        recommended_poste = _extract_recommended_poste(answer)
        years_experience = _extract_years_experience(answer)
        diploma_year = _extract_diploma_year(answer)

    elapsed = round(time.time() - t0, 2)
    displayed_poste = search_poste or recommended_poste or "Non précisé"

    return {
        "answer":            answer,
        "score":             score,
        "poste":             displayed_poste,
        "recommended_poste": recommended_poste or "Non précisé",
        "sources":           sources,
        "elapsed_seconds":   elapsed,
        "years_experience":  years_experience,
        "diploma_year":      diploma_year,
        "filename":          filename,
    }


# ─────────────────────────────────────────────────────────────
# Tri batch
# ─────────────────────────────────────────────────────────────

def sort_results_with_tiebreaker(results: list) -> list:
    def sort_key(r: dict) -> tuple:
        score = r.get("score")
        if score is None:
            score = _extract_score(r.get("answer", ""))
        score = score if score is not None else -1

        years_exp = r.get("years_experience")
        years_exp = years_exp if (years_exp is not None and years_exp >= 0) else -1

        diploma_year = r.get("diploma_year")
        diploma_year = diploma_year if (diploma_year and diploma_year > 0) else 9999

        return (-score, -years_exp, diploma_year)

    return sorted(results, key=sort_key)


# ─────────────────────────────────────────────────────────────
# Parsers de secours (texte libre / ancien format) — utilisés uniquement
# si le JSON n'a pas pu être récupéré après les 2 tentatives.
# ─────────────────────────────────────────────────────────────

def _extract_score(text: str) -> Optional[int]:
    patterns = [
        r"SCORE[^:\n]*:\s*\**\s*(\d{1,2})\s*\**\s*/\s*10",
        r"SCORE[^:\n]*:\s*\[?(\d{1,2})\]?\s*/\s*10",
        r"\*\*(\d{1,2})\*\*\s*/\s*10",
        r":\s*(\d{1,2})\s*/10",
        r"(\d{1,2})\s*/\s*10",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 10:
                return val
    m = re.search(r'\b(\d{1,2})/10\b', text)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val
    return None


def _extract_recommended_poste(text: str) -> Optional[str]:
    patterns = [
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]\s*([^\n\[\]]+)",
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*\n+\s*([^\n\[\]\*-][^\n\[\]\*]{3,})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("*•[] \t")
            if value and "[" not in value and len(value) > 3 and not _is_sentence_not_title(value):
                return value
    return None


def _extract_years_experience(text: str) -> Optional[int]:
    patterns = [
        r"ANN[EÉ]ES?[_\s]EXP[EÉ]RIENCE[^:\n]*:\s*\**\s*(-?\d{1,2})\s*\**",
        r"ANN[EÉ]ES?[_\s]EXP[^:\n]*:\s*(-?\d{1,2})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if -1 <= val <= 50:
                return val
    return None


def _extract_diploma_year(text: str) -> Optional[int]:
    patterns = [
        r"ANN[EÉ]E[_\s]DIPLOM[EÉ][^:\n]*:\s*\**\s*((?:19|20)\d{2})\s*\**",
        r"ANN[EÉ]E[_\s]DIPLOM[^:\n]*:\s*((?:19|20)\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None
