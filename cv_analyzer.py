"""
Module d'analyse de CV via pipeline RAG.
Extraction CV + recherche exigences + analyse LLM.
"""

import io
import logging
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
            raise RuntimeError("Installez : pip install pymupdf")


def extract_text_from_docx(content: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except ImportError:
        raise RuntimeError("Installez : pip install python-docx")


def extract_cv_text(content: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif name.endswith(".docx"):
        return extract_text_from_docx(content)
    elif name.endswith(".txt"):
        return content.decode("utf-8", errors="replace").strip()
    raise ValueError(f"Format non supporté : {filename}. Acceptés : PDF, DOCX, TXT")


# ─────────────────────────────────────────────────────────────
# Validation CV
# ─────────────────────────────────────────────────────────────

MIN_CV_LENGTH = 200


def validate_cv_text(cv_text: str, filename: str) -> Optional[dict]:
    if not cv_text or len(cv_text.strip()) < MIN_CV_LENGTH:
        return {
            "answer": (
                f"⚠️ CV insuffisant ({filename}) — {len(cv_text.strip())} caractères "
                f"(minimum : {MIN_CV_LENGTH}). Soumettez un CV complet."
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
# PROMPT STANDARD — poste précisé
# Court, structuré, calcul de score interne non affiché
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """Tu es un recruteur RH expert chez Sonatrach.

POSTE CIBLE : {poste}

EXIGENCES DU POSTE :
{job_context}

CV DU CANDIDAT :
{cv_text}

---
CALCUL INTERNE DU SCORE (ne pas afficher ce calcul) :
- Compatibilité domaine : 0-3 pts
- Diplôme/Formation : 0-2 pts
- Compétences techniques : 0-2 pts
- Expérience professionnelle : 0-2 pts
- Bonus Oil & Gas / Sonatrach : 0-1 pt
TOTAL : X/10

RÈGLES ABSOLUES :
- Si le domaine du CV est incompatible avec le poste → SCORE MAX 2/10
- Ne jamais inventer une compétence absente du CV
- Un diplôme seul sans expérience → max 6/10
- Les soft skills ne compensent pas les lacunes techniques

RÉPONDS UNIQUEMENT avec ce format, rien d'autre :

**SCORE** : X/10
**DOMAINE** : [domaine principal du candidat]
**DÉCISION** : Recommandé / À étudier / Non recommandé

**ATOUTS**
- [atout 1]
- [atout 2]
- [atout 3]

**LACUNES**
- [lacune 1]
- [lacune 2]

**POSTE RECOMMANDÉ** :
- [poste 1 adapté au vrai domaine du CV]
- [poste 2]
- [poste 3 si pertinent]

**ANNÉES_EXPÉRIENCE** : [nombre entier ou -1]
**ANNÉE_DIPLOME** : [année ou 0]
"""


# ─────────────────────────────────────────────────────────────
# PROMPT DÉCOUVERTE — sans poste précisé
# ─────────────────────────────────────────────────────────────

DISCOVERY_PROMPT = """Tu es un recruteur RH expert chez Sonatrach.

Un candidat soumet son CV sans poste visé. Identifie son profil et propose les meilleurs postes Sonatrach compatibles.

CV DU CANDIDAT :
{cv_text}

RÉFÉRENTIEL POSTES SONATRACH :
{job_context}

---
CALCUL INTERNE DU SCORE (ne pas afficher ce calcul) :
- Niveau de formation : 0-3 pts
- Expérience professionnelle réelle : 0-4 pts
- Pertinence pour Sonatrach : 0-2 pts
- Bonus Oil & Gas : 0-1 pt
TOTAL : X/10

RÈGLES :
- Identifier le domaine principal RÉEL du candidat
- Proposer 2-3 postes Sonatrach vraiment compatibles avec ce domaine
- Ne jamais proposer un poste informatique à un soudeur, ni un poste technique à un administratif

RÉPONDS UNIQUEMENT avec ce format, rien d'autre :

**SCORE** : X/10
**DOMAINE** : [domaine principal détecté]
**DÉCISION** : Recommandé / À étudier / Non recommandé

**ATOUTS**
- [atout 1]
- [atout 2]
- [atout 3]

**LACUNES**
- [lacune 1]
- [lacune 2]

**POSTE RECOMMANDÉ** :
- [poste 1 le plus adapté]
- [poste 2]
- [poste 3 si pertinent]

**ANNÉES_EXPÉRIENCE** : [nombre entier ou -1]
**ANNÉE_DIPLOME** : [année ou 0]
"""


def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    fallback = (
        "Aucun référentiel spécifique trouvé. "
        "Appliquer les critères RH généraux Sonatrach."
    )
    ctx = job_context or fallback

    if not poste:
        return DISCOVERY_PROMPT.format(cv_text=cv_text[:4000], job_context=ctx)

    return ANALYSIS_PROMPT.format(
        poste=poste,
        job_context=ctx,
        cv_text=cv_text[:4000],
    )


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
        logger.warning(
            "CV '%s' rejeté : contenu insuffisant (%d caractères)",
            filename, len(cv_text.strip())
        )
        return validation_error

    search_poste = poste.strip() if poste else ""

    if search_poste:
        search_query = f"exigences compétences diplômes requis poste {search_poste}"
    else:
        cv_hint = " ".join(cv_text[:600].split())[:300]
        search_query = f"postes Sonatrach compatibles profil {cv_hint}"

    # ── Recherche RAG ──────────────────────────────────────────
    try:
        query_embedding = pipeline.embedder.embed_single(search_query)

        dense_results = pipeline.vector_store.search(
            query_embedding,
            k=pipeline.config.top_k_dense,
        )

        sparse_results = []
        if pipeline.bm25:
            sparse_results = pipeline.bm25.search(
                search_query,
                k=pipeline.config.top_k_sparse,
            )

        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=pipeline.config.rrf_k,
        )

        top_chunks = fused[:pipeline.config.top_k_after_rerank]

        if pipeline.reranker and top_chunks:
            pairs = [(search_query, c["content"]) for c in top_chunks]
            scores = pipeline.reranker.model.predict(pairs)
            for c, s in zip(top_chunks, scores):
                c["rerank_score"] = float(s)
            top_chunks = sorted(
                top_chunks,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True,
            )

        job_context = "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', '?')}]\n{c['content']}"
            for c in top_chunks
        )
        sources = list({c["metadata"].get("source", "?") for c in top_chunks})

    except Exception as e:
        logger.warning("Erreur recherche RAG : %s", e)
        job_context = ""
        sources = []

    # ── Appel LLM ─────────────────────────────────────────────
    prompt = build_analysis_prompt(
        cv_text=cv_text,
        poste=search_poste,
        job_context=job_context,
    )

    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system=(
                "Tu es un expert RH chez Sonatrach. "
                "Réponds UNIQUEMENT en français avec le format exact demandé. "
                "N'affiche JAMAIS le barème ni les points par critère. "
                "Sois concis : maximum 10 lignes hors format obligatoire."
            ),
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur analyse LLM : {e}")

    score             = _extract_score(answer)
    recommended_poste = _extract_recommended_poste(answer)
    years_experience  = _extract_years_experience(answer)
    diploma_year      = _extract_diploma_year(answer)

    if score is None:
        logger.warning(
            "Score non extrait pour '%s'. Début réponse LLM :\n%s",
            filename, answer[:500]
        )

    elapsed = round(time.time() - t0, 2)

    return {
        "answer":            answer,
        "score":             score,
        "poste":             search_poste or recommended_poste or "Non précisé",
        "recommended_poste": recommended_poste or "Non précisé",
        "sources":           sources,
        "elapsed_seconds":   elapsed,
        "years_experience":  years_experience,
        "diploma_year":      diploma_year,
        "filename":          filename,
    }


# ─────────────────────────────────────────────────────────────
# Tri batch avec départage
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
# Parsers
# ─────────────────────────────────────────────────────────────

def _extract_score(text: str) -> Optional[int]:
    import re

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
    import re

    # Multi-lignes : bloc après "POSTE RECOMMANDÉ :" jusqu'au prochain label **
    m = re.search(
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]?\s*\n([\s\S]*?)"
        r"(?=\n\s*\*\*[A-ZÀÂÉÈÊËÎÏÔÙÛÜ_\s]+\*\*|\Z)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if m:
        block = m.group(1)
        lines = []
        for line in block.splitlines():
            cleaned = line.strip().lstrip("-•* \t")
            if (
                cleaned
                and len(cleaned) > 3
                and not cleaned.upper().startswith("ANNÉES")
                and not cleaned.upper().startswith("ANNÉE_")
                and not cleaned.startswith("**")
                and "[" not in cleaned
            ):
                lines.append(cleaned)
        if lines:
            return " / ".join(lines)

    # Fallback : poste unique sur la même ligne
    patterns = [
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]\s*([^\n\[\]]+)",
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*\n+\s*([^\n\[\]\*]{4,})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("*•-[] \t")
            if value and len(value) > 3 and "[" not in value:
                return value

    return None


def _extract_years_experience(text: str) -> Optional[int]:
    import re
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
    import re
    patterns = [
        r"ANN[EÉ]E[_\s]DIPLOM[EÉ][^:\n]*:\s*\**\s*((?:19|20)\d{2})\s*\**",
        r"ANN[EÉ]E[_\s]DIPLOM[^:\n]*:\s*((?:19|20)\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None
